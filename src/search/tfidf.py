import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSearch:
    def __init__(self):
        print('loading tfidf')
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "tfidf"))
        self.vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))
        self.tfidf_matrix = joblib.load(os.path.join(base_dir, "tfidf_matrix.pkl"))
        self.doc_ids = joblib.load(os.path.join(base_dir, "tfidf_doc_ids.pkl"))
        print('loaded tfidf')

    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(self.doc_ids[idx])
        return results