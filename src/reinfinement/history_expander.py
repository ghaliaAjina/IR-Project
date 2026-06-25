from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class HistoryQueryExpander:

    def __init__(self, user_history, top_k=5):
        self.user_history = user_history
        self.top_k = top_k

    def expand(self, user_id):
        history = self.user_history.get(user_id, [])
        if not history:
            return []
        words = []
        for q in history:
            for word in q.lower().split():
                if word not in ENGLISH_STOP_WORDS:
                    words.append(word)
        freq = Counter(words)
        return [
            word
            for word, _ in freq.most_common(self.top_k)
        ]

