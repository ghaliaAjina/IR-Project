from history_expander import HistoryQueryExpander
from tfidf_expander import TFIDFQueryExpander
from semantic_expander import SemanticQueryExpander
from query_refiner import QueryRefiner


documents = [
    "python machine learning algorithms",
    "information retrieval bm25 ranking",
    "deep learning neural networks",
    "bert semantic search models"
]


user_history = {

    1: [
        "bm25 search",
        "information retrieval",
        "ranking algorithms"
    ]

}


history = HistoryQueryExpander(
    user_history
)

tfidf = TFIDFQueryExpander(
    documents
)

semantic = SemanticQueryExpander(
    documents
)


query_refiner = QueryRefiner(
    history,
    tfidf,
    semantic
)


query = query_refiner.refine(
    user_id=1,
    query="search algorithms"
)


print(query)