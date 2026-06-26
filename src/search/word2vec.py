import os

import joblib
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Word2VecSearch:
    def __init__(self):
        print('loading word2vec')
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "word2vec"))
        self.model = Word2Vec.load(os.path.join(base_dir, "word2vec_model.model"))
        self.doc_vectors = joblib.load(os.path.join(base_dir, "word2vec_doc_vectors.pkl"))
        self.doc_ids = joblib.load(os.path.join(base_dir, "word2vec_doc_ids.pkl"))
        print('loaded word2vec')

    def query_vector(self, text):
        tokens = text.split()
        vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]
        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

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
