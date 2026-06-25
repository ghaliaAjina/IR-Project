import os
import pickle
from collections import defaultdict
import math

import mysql.connector


class Evaluator:
    def __init__(self, tfidf, word2vec, bm25, bert, hybridParallelBM25Word2Vec, hybridSerialBM25BERT):
        self.tfidf_search = tfidf
        self.word2vec_search = word2vec
        self.bm25_search = bm25
        self.bert_search = bert
        self.hybrid_parallel = hybridParallelBM25Word2Vec
        self.hybrid_serial = hybridSerialBM25BERT
        self.results = None

    # =====================================================================
    # 2. Mathematical Metric Calculations
    # =====================================================================

    def compute_query_metrics(self, ranked_list: list, qrel_list: list, top_k_p: int = 10) -> dict:
        """
        Computes Precision@K, Recall, Average Precision (for MAP), and nDCG 
        for a single query sequence.
        """
        # Total actual relevant documents in the ground truth for this query (relevance > 0)
        total_relevant = sum(1 for rel in qrel_list)
        
        if total_relevant == 0:
            return None # Skip queries that have no true positive judgments

        # 1. Precision @ K (e.g., P@10)
        top_k_docs = ranked_list[:top_k_p]
        rel_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in qrel_list)
        precision_at_k = rel_in_top_k / top_k_p

        # 2. Recall (evaluated across the total retrieved ranked list)
        rel_retrieved = sum(1 for doc_id in ranked_list if doc_id in qrel_list)
        recall = rel_retrieved / total_relevant

        # 3. Average Precision (AP) -> used to calculate Mean Average Precision (MAP)
        ap_sum = 0.0
        num_rel_found = 0
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in qrel_list:
                num_rel_found += 1
                precision_at_rank = num_rel_found / rank
                ap_sum += precision_at_rank
        ap = ap_sum / total_relevant

        # 4. Normalized Discounted Cumulative Gain (nDCG)
        # Calculate Discounted Cumulative Gain (DCG)
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_list, start=1):
            rel = 1 if doc_id in qrel_list else 0
            dcg += (2**rel - 1) / math.log2(rank + 1)

        # Calculate Ideal DCG (IDCG) based on the best possible sorting of true judgments
        ideal_rels = sorted([1 if doc_id in qrel_list else 0 for doc_id in ranked_list], reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:len(ranked_list)], start=1):
            idcg += (2**rel - 1) / math.log2(rank + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "p_at_k": precision_at_k,
            "recall": recall,
            "ap": ap,
            "ndcg": ndcg
        }
    
    def evaluate(self):
        self.results = {}
        cache_file = os.path.join(os.path.dirname(__file__), "evaluation_results.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.results = pickle.load(f)
                print(f"Loaded cached evaluation results from {cache_file}")
                return self.results
            except (pickle.PickleError, EOFError, OSError):
                pass

        db_config = {
            "host": "localhost",
            "user": "root",           
            "password": "1234",  
            "database": "ir_dataset"
        }   
        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            # --- Step A: Load Ground Truth (qrels) into a fast lookup dictionary ---
            print("Loading qrels from database...")
            cursor.execute("SELECT query_id, doc_id, relevance FROM qrels")
            
            # Nested structure: qrels_lookup[query_id][doc_id] = relevance_score
            qrels_lookup = defaultdict(list)
            for row in cursor.fetchall():
                qrels_lookup[row['query_id']].append(row['doc_id'])

            # --- Step B: Load Preprocessed Queries ---
            print("Loading preprocessed queries...")
            cursor.execute("SELECT query_id, text_clean FROM preprocessed_queries")
            queries = cursor.fetchall()
            
            print(f"Loaded {len(queries)} queries and {len(qrels_lookup)} qrel mappings.\n")

            # --- Step C: Evaluate Models Loop ---
            models_to_evaluate = {
                "TF-IDF": self.tfidf_search.search,
                "Word2Vec": self.word2vec_search.search,
                "BM25": self.bm25_search.search,
                "BERT": self.bert_search.search,
                "Parallel (BM25+W2V)": self.hybrid_parallel.search,
                "Serial (BM25→BERT)": self.hybrid_serial.search
            }

            # Dictionary to house aggregated tracking data
            final_results = {}

            for model_name, retrieve_fn in models_to_evaluate.items():
                print(f"Running evaluation for model: {model_name}...")
                
                total_p_at_10 = 0.0
                total_recall = 0.0
                total_ap = 0.0
                total_ndcg = 0.0
                evaluated_queries_count = 0

                for q in queries:
                    q_id = q['query_id']
                    q_text = q['text_clean']

                    # # Skip if this query text is completely blank or missing ground truth
                    # if not q_text or q_id not in qrels_lookup:
                    #     continue

                    # Call the model stub to get back ranked array of doc IDs (Top 100 documents)
                    ranked_results = retrieve_fn(query=q_text, top_k=100)
                    
                    # If the stub returned None or empty array, treat metrics as 0 for this query
                    if not ranked_results:
                        ranked_results = []

                    # Compute individual query metrics
                    metrics = self.compute_query_metrics(ranked_results, qrels_lookup[q_id], top_k_p=10)
                    
                    if metrics:
                        total_p_at_10 += metrics["p_at_k"]
                        total_recall += metrics["recall"]
                        total_ap += metrics["ap"]
                        total_ndcg += metrics["ndcg"]
                        evaluated_queries_count += 1

                # Calculate means across the entire dataset split
                if evaluated_queries_count > 0:
                    final_results[model_name] = {
                        "MAP": total_ap / evaluated_queries_count,
                        "Recall": total_recall / evaluated_queries_count,
                        "Precision@10": total_p_at_10 / evaluated_queries_count,
                        "nDCG": total_ndcg / evaluated_queries_count
                    }
                else:
                    final_results[model_name] = {"MAP": 0.0, "Recall": 0.0, "Precision@10": 0.0, "nDCG": 0.0}

            # =====================================================================
            # 4. Print Summary Comparison Table
            # =====================================================================
            print("\n======================= EVALUATION REPORT =======================")
            print(f"{'Model':<20} | {'MAP':<10} | {'Recall':<10} | {'Precision@10':<12} | {'nDCG':<10}")
            print("-" * 75)
            for model_name, metrics in final_results.items():
                print(f"{model_name:<20} | {metrics['MAP']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['Precision@10']:<12.4f} | {metrics['nDCG']:<10.4f}")
            print("===================================================================")

            # Save results object on the Evaluator instance for later use
            self.results = final_results
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(final_results, f)
            except OSError as e:
                print(f"Failed to save evaluation cache: {e}")

            # Also return the results object
            return final_results

        except mysql.connector.Error as err:
            print(f"MySQL Error: {err}")

        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()
