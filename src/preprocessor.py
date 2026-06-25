import re

import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self, remove_stopwords=True, stemming=True):
        nltk.download('stopwords')
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str):
        return text.split()

    def remove_stopwords_fn(self, tokens):
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens):
        if not self.stemming:
            return tokens
        return [self.stemmer.stem(t) for t in tokens]

    def process(self, text: str):
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords_fn(tokens)
        tokens = self.stem(tokens)
        return tokens