from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticQueryExpander:

    def __init__(self, corpus, top_k=5):
        self.top_k = top_k
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.corpus = corpus
        self.embeddings = self.model.encode(
            corpus,
            normalize_embeddings=True
        )

    def expand(self, query):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        scores = np.dot(
            self.embeddings,
            query_embedding
        )
        indexes = np.argsort(
            scores
        )[::-1][:self.top_k]


        return [
            self.corpus[i]
            for i in indexes
        ]
