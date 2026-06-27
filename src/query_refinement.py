
import json
import os
import nltk
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker

from preprocessor import Preprocessor


class QueryRefiner:

    STOP_EXPANSION = {
    "world","war","history","time","day","year","part",
    "what","which","where","when","who","why","how",
    "can","is","are","the","difference","same","good","bad"
}

    def __init__(self, profile_file="user_profiles.json"):
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        self.preprocessor = Preprocessor()
        self.spell = SpellChecker()

        self.profile_file = profile_file

        if not os.path.exists(profile_file):
            with open(profile_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_profiles(self):
        with open(self.profile_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_profiles(self, profiles):
        with open(self.profile_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4)

    def update_user_profile(self, user_id, corrected_query):
        profiles = self._load_profiles()
        history = profiles.get(user_id, [])
        history.append(corrected_query)
        history = history[-20:]
        profiles[user_id] = history
        self._save_profiles(profiles)


    def spelling_correction(self, query):
        corrected = []
        for word in query.split():
            corrected.append(self.spell.correction(word) or word)

        return " ".join(corrected)



    def synonym_expansion(self, query):
       query_lower = query.lower()
       tokens = query_lower.split()
       expanded = query.split()

       for token in tokens[:2]:
        if (
            token in self.STOP_EXPANSION
            or token.isdigit()
            or not token.isalpha()
        ):
            continue
        synonyms = []

        for syn in wn.synsets(token):
            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ").lower().strip()
                if (
                    word != token
                    and len(word.split()) == 1
                    and word not in expanded
                    and word not in synonyms
                ):
                    synonyms.append(word)

        if synonyms:
            expanded.append(synonyms[0])

        return " ".join(dict.fromkeys(expanded))

    def personalize(self, query, user_id="default"):
        profiles = self._load_profiles()
        history = profiles.get(user_id, [])
        if not history:
            return query
        freq = {}
        for q in history:
            for w in q.lower().split():
                if len(w) <= 2:
                    continue
                freq[w] = freq.get(w, 0) + 1

        top_words = sorted(freq, key=freq.get, reverse=True)[:3]
        final = query

        for w in top_words:
            if w not in final.lower():
                final += " " + w

        return final

    def refine(self, query, user_id="default"):
        corrected = self.spelling_correction(query)
        expanded = self.synonym_expansion(corrected)
        personalized = self.personalize(expanded, user_id)
        tokens = self.preprocessor.process(personalized)
        self.update_user_profile(user_id, corrected)
        return {
            "original_query": query,
            "corrected_query": corrected,
            "expanded_query": expanded,
            "personalized_query": personalized,
            "processed_query": tokens,
            "processed_query_text": " ".join(tokens)
        }

if __name__ == "__main__":
    qr = QueryRefiner()
    while True:
        q = input("Query: ")
        if q.lower() == "exit":
            break
        result = qr.refine(q)
        print("\n========== RESULT ==========")
        for k, v in result.items():
            print(f"{k}: {v}")
