import os

from flask import Flask, jsonify, render_template, request

from db import DB
from query_refinement import QueryRefiner
from search.bert import BertSearch
from search.bm25 import BM25Search
from search.hybrid_search import HybridParallelBM25Word2Vec, HybridSerialBM25BERT
from search.tfidf import TFIDFSearch
from search.word2vec import Word2VecSearch
from models import (
     bm25,
    bert,
    word2vec,
    tfidf,
    hybrid_serial,
    hybrid_parallel,
)
from evaluator import Evaluator


def format_results(results):
    if results is None:
        return "No results returned."
    if isinstance(results, dict):
        return results
    if isinstance(results, list):
        return [str(item) for item in results]
    return [str(results)]

template_dir = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=template_dir, static_folder=template_dir, static_url_path='/static')
services = {
    "bert": bert,
    "bm25": bm25,
    "tfidf": tfidf,
    "hybrid_serial_search": hybrid_serial,
    "hybrid_parallel_search": hybrid_parallel,
    "word2vec": word2vec,
}
query_refiner = QueryRefiner()
db = DB()
evaluator = Evaluator(tfidf=tfidf, word2vec=word2vec, bm25=bm25, bert=bert, hybridParallelBM25Word2Vec=hybrid_parallel, hybridSerialBM25BERT=hybrid_serial)


@app.route("/")
def index():
    return render_template("index.html", services=services.keys())


@app.route("/search", methods=["POST"])
def search_route():
    service_name = request.form.get("service")
    query = request.form.get("query", "").strip()
    top_k = int(request.form.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    service = services.get(service_name)
    if service is None:
        return jsonify({"error": f"Unknown service: {service_name}"}), 400

    try:
        refinement = query_refiner.refine(
            query=query,
            user_id="default"
        )
        print('refinement query:', refinement)
        
        search_kwargs = {"top_k": top_k}
        if service_name == "hybrid_parallel_search":
            alpha = float(request.form.get("alpha", 0.5))
            beta = float(request.form.get("beta", 0.5))
            
            weight_sum = round(alpha + beta, 2)
            if abs(weight_sum - 1.0) > 0.01:
                return jsonify({"error": f"Alpha ({alpha}) and Beta ({beta}) must sum to 1.0, but sum to {weight_sum}"}), 400
            
            search_kwargs["alpha"] = alpha
            search_kwargs["beta"] = beta
        
        doc_ids = service.search(refinement["processed_query_text"], **search_kwargs)
        query_refiner.update_user_profile("default",refinement["corrected_query"])
        docs_by_id = db.get_documents_by_ids(doc_ids)
        results_with_text = [
            {"doc_id": doc_id, "text": docs_by_id.get(doc_id, "")}
            for doc_id in doc_ids
        ]
        return jsonify({
            "query_refinement": refinement,
            "results": results_with_text,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/evaluate", methods=["GET"])
def evaluate_route():
    try:
        results = evaluator.evaluate()
        return jsonify(results)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

