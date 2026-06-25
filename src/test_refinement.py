"""
Test script for query refinement module
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from reinfinement import (
    HistoryQueryExpander,
    TFIDFQueryExpander,
    SemanticQueryExpander,
    QueryRefiner
)


def test_query_refinement():
    """Test the complete query refinement pipeline"""
    
    # Sample documents for indexing
    documents = [
        "python machine learning algorithms",
        "information retrieval bm25 ranking",
        "deep learning neural networks",
        "bert semantic search models"
    ]

    # Sample user history
    user_history = {
        1: [
            "bm25 search",
            "information retrieval",
            "ranking algorithms"
        ]
    }

    # Initialize expanders
    print("Initializing expanders...")
    history = HistoryQueryExpander(user_history)
    tfidf = TFIDFQueryExpander(documents)
    semantic = SemanticQueryExpander(documents)

    # Create query refiner
    query_refiner = QueryRefiner(history, tfidf, semantic)

    # Test refinement
    original_query = "search algorithms"
    print(f"\nOriginal Query: {original_query}")
    
    refined_query = query_refiner.refine(user_id=1, query=original_query)
    print(f"Refined Query: {refined_query}")
    
    # Test with different user
    print("\n" + "="*50)
    refined_query_2 = query_refiner.refine(user_id=2, query="learning models")
    print(f"Original Query: learning models")
    print(f"Refined Query: {refined_query_2}")
    
    print("\n" + "="*50)
    print("✓ Query refinement test completed successfully!")


if __name__ == "__main__":
    test_query_refinement()
