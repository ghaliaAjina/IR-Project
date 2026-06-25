import numpy as np

from search.bert import BertSearch
from search.bm25 import BM25Search
from search.word2vec import Word2VecSearch
from sklearn.metrics.pairwise import cosine_similarity


class HybridParallelBM25Word2Vec:
    def __init__(self, bm25_search, word2vec_search):
        print('loading hybrid parallel')
        self.bm25_search = bm25_search
        self.word2vec_search = word2vec_search
        self.doc_ids = self.bm25_search.doc_ids
    
    def search(self, query, top_k=10, alpha=0.5, beta=0.5):
        """
        Retrieve using both BM25 and Word2Vec, combine scores, return top-k
        """
        # Get results and scores from both models
        bm25_tokens = query.lower().split()
        bm25_scores = self.bm25_search.bm25.get_scores(bm25_tokens)
        
        # Word2Vec scores
        w2v_query_vec = self.word2vec_search.query_vector(query)
        w2v_scores = cosine_similarity(
            [w2v_query_vec],
            self.word2vec_search.doc_vectors
        )[0]
        
        # Normalize scores to [0, 1]
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        w2v_scores_norm = (w2v_scores - w2v_scores.min()) / (w2v_scores.max() - w2v_scores.min() + 1e-10)
        
        # Combine with equal weights
        combined_scores = alpha * bm25_scores_norm + beta * w2v_scores_norm
        
        # Get top-k results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [self.doc_ids[idx] for idx in top_indices]
        
        return results


class HybridSerialBM25BERT:
    """
    Serial Hybrid: BM25 retrieval followed by BERT re-ranking
    First stage: BM25 retrieves candidate documents
    Second stage: BERT re-ranks the candidates
    """
    def __init__(self, bm25_search, bert_search, candidate_k=100):
        print('loading hybrid sreial')
        self.bm25_search = bm25_search
        self.bert_search = bert_search
        self.candidate_k = candidate_k  # Number of candidates from BM25
        self.doc_ids = self.bm25_search.doc_ids
    
    def search(self, query, top_k=10):
        """
        Retrieve using BM25, then re-rank with BERT
        """
        # Stage 1: BM25 retrieval to get candidates
        bm25_candidates = self.bm25_search.search(query, top_k=self.candidate_k)
        
        # Stage 2: BERT re-ranking on candidates
        query_embedding = self.bert_search.query_vector(query)
        
        # Get embeddings for only the candidate documents
        candidate_indices = [self.doc_ids.index(doc_id) for doc_id in bm25_candidates]
        candidate_embeddings = self.bert_search.doc_vectors[candidate_indices]
        
        # Calculate BERT scores for candidates
        bert_scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Sort by BERT scores and return top-k
        sorted_indices = np.argsort(bert_scores)[::-1][:top_k]
        results = [bm25_candidates[idx] for idx in sorted_indices]
        
        return results
