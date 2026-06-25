import os
import joblib


class BM25Search:
    def __init__(self):
        print('loading bm25')
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "bm25"))
        self.bm25 = joblib.load(os.path.join(base_dir, "bm25_model.pkl"))
        self.doc_ids = joblib.load(os.path.join(base_dir, "bm25_doc_ids.pkl"))
        print('loaded bm25')

    def search(self, query, top_k=10):
        tokens = query.lower().split() 
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(self.doc_ids[idx])
        return results
