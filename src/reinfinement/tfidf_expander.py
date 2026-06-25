from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFQueryExpander:

    def __init__(self, corpus, top_k=5):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(
            corpus
        )
        self.words = self.vectorizer.get_feature_names_out()

    def expand(self, query):
        query_vector = self.vectorizer.transform(
            [query]
        )
        scores = query_vector.toarray()[0]
        ranked = sorted(
            zip(self.words, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            word
            for word, score in ranked[:self.top_k]
            if score > 0
        ]
