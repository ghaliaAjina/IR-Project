import re

import nltk
from nltk.corpus import wordnet as wn

from preprocessor import Preprocessor


class QueryRefiner:
    def __init__(self):
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        self.preprocessor = Preprocessor()

    def fix_grammar(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+([?.!,;:])", r"\1", text)
        text = re.sub(r"([?.!,;:])([^\s])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text)
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in ".?!":
                text = text + "?"
        return text

    def _get_synonyms(self, token: str) -> list[str]:
        synonyms = set()
        for synset in wn.synsets(token):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != token.lower() and " " not in synonym:
                    synonyms.add(synonym.lower())
            if len(synonyms) >= 4:
                break
        return list(synonyms)

    def add_synonyms(self, text: str) -> str:
        tokens = re.findall(r"\b\w+\b", text.lower())
        stop_tokens = {
            "the", "a", "an", "and", "or", "but", "with", "for", "to", "of",
            "in", "on", "at", "by", "from", "is", "are", "was", "were",
            "this", "that", "these", "those", "it", "its",
        }

        synonyms_to_add = []
        seen = set()
        for token in tokens:
            if token in stop_tokens or token.isdigit() or len(token) < 3:
                continue
            for synonym in self._get_synonyms(token):
                if synonym not in seen and synonym != token:
                    synonyms_to_add.append(synonym)
                    seen.add(synonym)
                    break
            if len(synonyms_to_add) >= 3:
                break

        if synonyms_to_add:
            return f"{text} {' '.join(synonyms_to_add)}"
        return text

    def refine(self, text: str) -> dict[str, object]:
        fixed = self.fix_grammar(text)
        expanded = self.add_synonyms(fixed)
        processed_tokens = self.preprocessor.process(expanded)
        processed_query_text = " ".join(processed_tokens)
        return {
            "original_query": text,
            "refined_query": expanded,
            "processed_query": processed_tokens,
            "processed_query_text": processed_query_text,
        }
