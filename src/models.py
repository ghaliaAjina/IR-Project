

from search.bert import BertSearch
from search.bm25 import BM25Search
from search.hybrid_search import HybridParallelBM25Word2Vec, HybridSerialBM25BERT
from search.tfidf import TFIDFSearch
from search.word2vec import Word2VecSearch


bm25 = BM25Search()
bert = BertSearch()
word2vec = Word2VecSearch()
tfidf = TFIDFSearch()

hybrid_serial = HybridSerialBM25BERT(
    bm25_search=bm25,
    bert_search=bert
)

hybrid_parallel = HybridParallelBM25Word2Vec(
    bm25_search=bm25,
    word2vec_search=word2vec
)