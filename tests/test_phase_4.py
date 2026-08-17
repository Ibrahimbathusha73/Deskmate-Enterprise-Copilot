import pytest
import time
import redis
from unittest.mock import patch, MagicMock

from ops.pii_redact import redact
from ops.semantic_cache import get_cached, set_cache
from agents.docs_rag_agent import docs_rag_agent, is_unanswerable

def test_pii_redaction():
    text = "Please reach out to me at john.doe@example.com or 123-456-7890. My SSN is 000-12-3456."
    redacted = redact(text)
    
    # Assert redactions occurred
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_SSN]" in redacted
    
    # Assert original sensitive values are gone
    assert "john.doe@example.com" not in redacted
    assert "123-456-7890" not in redacted
    assert "000-12-3456" not in redacted

def test_is_unanswerable():
    assert is_unanswerable("The context does not contain information about this.") is True
    assert is_unanswerable("I cannot answer this question based on the provided context.") is True
    assert is_unanswerable("According to the context, the status is active.") is False

@patch("agents.docs_rag_agent.client")
def test_model_tiering_fallback(mock_client):
    # Mock response for openai/gpt-oss-20b indicating it cannot answer
    mock_resp_8b = MagicMock()
    mock_resp_8b.choices[0].message.content = "I do not have enough information to answer this query."
    mock_resp_8b.usage.prompt_tokens = 50
    mock_resp_8b.usage.completion_tokens = 15
    
    # Mock response for openai/gpt-oss-120b
    mock_resp_70b = MagicMock()
    mock_resp_70b.choices[0].message.content = "Detailed fallback answer from 70B."
    mock_resp_70b.usage.prompt_tokens = 100
    mock_resp_70b.usage.completion_tokens = 30

    # Configure client mock to return 8B first, then 70B
    mock_client.chat.completions.create.side_effect = [mock_resp_8b, mock_resp_70b]

    # Patch hybrid_search to return some dummy context chunks so retrieval passes
    with patch("agents.docs_rag_agent.hybrid_search", return_value=[{"id": "doc1", "text": "Some text", "score": -2.0}]), \
         patch("agents.docs_rag_agent.get_cached", return_value=None), \
         patch("agents.docs_rag_agent.set_cache") as mock_set_cache:
         
        res = docs_rag_agent("Test query")
        
        # Verify 70B response was returned
        assert res["answer"] == "Detailed fallback answer from 70B."
        
        # Verify client completions was called twice (8B, then 70B)
        assert mock_client.chat.completions.create.call_count == 2
        
        # Verify the model parameters passed
        calls = mock_client.chat.completions.create.call_args_list
        assert calls[0][1]["model"] == "openai/gpt-oss-20b"
        assert calls[1][1]["model"] == "openai/gpt-oss-120b"

def test_semantic_caching_integration():
    # Verify local Redis connectivity
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
    except Exception as e:
        pytest.skip(f"Redis is not running locally: {e}")
        
    # Clear cache database before test
    r.flushdb()
    
    # Run a real agent query that we know is answerable from our hybrid search issues DB
    query = "What is the pipeline device validation enhancement?"
    
    # First call: Cache Miss
    t0 = time.time()
    res1 = docs_rag_agent(query)
    duration1 = time.time() - t0
    
    # Second call: Cache Hit (exact match)
    t1 = time.time()
    res2 = docs_rag_agent(query)
    duration2 = time.time() - t1
    
    # Third call: Cache Hit (semantic match - slightly modified query)
    query_modified = "tell me about pipeline device validation enhancement"
    t2 = time.time()
    res3 = docs_rag_agent(query_modified)
    duration3 = time.time() - t2
    
    print(f"\nFirst call duration: {duration1:.4f}s (Cache Miss)")
    print(f"Second call duration: {duration2:.4f}s (Exact Cache Hit)")
    print(f"Third call duration: {duration3:.4f}s (Semantic Cache Hit)")
    
    # Assert cache hit latency is much smaller
    assert duration2 < duration1
    assert duration3 < duration1
    
    # Assert answer content is retrieved from cache correctly
    assert res2["answer"] == res1["answer"]
    assert res3["answer"] == res1["answer"]
