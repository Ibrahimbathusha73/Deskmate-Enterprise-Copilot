import pytest
import chromadb
from retrieval.hybrid_search import hybrid_search, collection

def test_collection_not_empty():
    """Verify that the ChromaDB collection contains ingested documents."""
    count = collection.count()
    assert count > 0, "ChromaDB collection should have at least one document."

def test_hybrid_search_queries():
    """Verify that hybrid search returns non-empty, relevant results for sample queries."""
    sample_queries = [
        "How to use pipeline for inference?",
        "tokenizer error or tokenization warning",
        "CUDA out of memory or GPU performance"
    ]
    
    for query in sample_queries:
        results = hybrid_search(query, top_k=5, final_k=3)
        assert isinstance(results, list), "Search results should be a list."
        assert len(results) > 0, f"Query '{query}' should return at least one result."
        
        # Check that result elements contain the expected fields
        for doc in results:
            assert "id" in doc
            assert "text" in doc
            assert "score" in doc
            assert isinstance(doc["score"], float)
