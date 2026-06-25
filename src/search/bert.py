import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class BertSearch:

    def __init__(self):
        print('loading bert')
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "bert"))
        self.doc_vectors = joblib.load(
            os.path.join(base_dir, "bert_doc_vectors.pkl")
        )
        self.doc_ids = joblib.load(
            os.path.join(base_dir, "bert_doc_ids.pkl")
        )
        print('loaded bert')

    # -------------------------
    # Query vector
    # -------------------------
    def query_vector(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    # -------------------------
    # Search
    # -------------------------
    def search(self, query, top_k=10):
        qvec = self.query_vector(query)
        scores = cosine_similarity(
            [qvec],
            self.doc_vectors
        )[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(self.doc_ids[idx])
        return results