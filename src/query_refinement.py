
import json
import os
import nltk
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker

from preprocessor import Preprocessor


class QueryRefiner:
    """
    Query Refinement Service
    ------------------------
    1. Spelling Correction
    2. Domain-aware Query Expansion (LoTTE Lifestyle)
    3. WordNet Fallback
    4. User Personalization
    5. Preprocessing
    """

    DOMAIN_SYNONYMS = {
    # Music
    "guitar": ["acoustic", "electric", "strings", "tuner", "capo", "frets", "ukulele"],
    "ukulele": ["guitar", "strings", "tuner"],
    "bass": ["guitar", "strings"],
    "music": ["notes", "tempo", "rhythm", "beat", "scale", "key", "octave"],
    "tempo": ["bpm", "beat", "rhythm"],
    "beat": ["tempo", "rhythm"],
    "scale": ["mode", "key", "notes"],
    "key": ["scale", "notes"],
    "octave": ["notes", "pitch"],
    "piano": ["keyboard", "music"],
    "drums": ["drumsticks", "rhythm"],

    # Coffee
    "coffee": ["espresso", "americano", "latte", "cappuccino", "beans", "grinder", "brew"],
    "espresso": ["coffee", "beans"],

    # Pets
    "dog": ["puppy", "breed", "feeding"],
    "mastiff": ["dog", "puppy"],
    "cat": ["kitten", "feline"],
    "rabbit": ["bunny", "pet"],
    "fish": ["aquarium", "tank"],
    "aquarium": ["fish", "tank", "filter", "shrimp"],
    "shrimp": ["aquarium", "fish"],
    "snake": ["reptile"],
    "betta": ["fish", "aquarium"],
    "goldfish": ["fish", "aquarium"],

    # Cycling
    "bike": ["bicycle", "cycling", "tire", "chain", "wheel", "brake"],
    "bicycle": ["bike", "cycling"],
    "cycling": ["bike", "bicycle"],

    # Cars
    "car": ["engine", "battery", "radiator", "brake", "coolant", "transmission"],
    "engine": ["car", "motor"],
    "battery": ["car", "voltage"],
    "radiator": ["coolant", "engine"],

    # Gardening
    "garden": ["plants", "soil", "watering", "fertilizer"],
    "plant": ["garden", "soil"],
    "tomato": ["garden", "plants"],

    # DIY
    "paint": ["primer", "latex", "acrylic"],
    "wood": ["timber"],
    "garage": ["door"],
    "door": ["garage"],
}

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
        expanded = query.split()
        for phrase, syns in self.DOMAIN_SYNONYMS.items():
            if phrase in query_lower:
                for s in syns[:2]:
                    if s not in expanded:
                        expanded.append(s)
                return " ".join(expanded)


        for token in query_lower.split()[:2]:
            if token in self.STOP_EXPANSION or token.isdigit():
                continue
            synsets = wn.synsets(token)
            if not synsets:
                continue
            syn = synsets[0]

            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if (
                    word != token
                    and len(word.split()) == 1
                    and word not in expanded
                ):
                    expanded.append(word)
                    break

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
