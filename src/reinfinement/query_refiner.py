class QueryRefiner:

    def __init__(
        self,
        history_expander,
        tfidf_expander,
        semantic_expander
    ):
        self.history = history_expander
        self.tfidf = tfidf_expander
        self.semantic = semantic_expander


    def refine(
        self,
        user_id,
        query
    ):
        terms = []
        terms.extend(
            self.history.expand(user_id)
        )
        terms.extend(
            self.tfidf.expand(query)
        )
        terms.extend(
            self.semantic.expand(query)
        )
        refined = (
            query
            + " "
            + " ".join(terms)
        )
        return refined

