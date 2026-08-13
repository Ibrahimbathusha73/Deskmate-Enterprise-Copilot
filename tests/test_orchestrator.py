import pytest
from orchestrator.graph import athena_graph

def test_docs_route():
    query = "What is the pipeline device validation enhancement?"
    state = athena_graph.invoke({"query": query})
    
    assert state["intent"] == "docs_question"
    assert "answer" in state and state["answer"] is not None
    assert "device" in state["answer"].lower() or "pipeline" in state["answer"].lower()
    assert state["retrieved_chunks"] is not None
    assert len(state["retrieved_chunks"]) > 0
    assert state["needs_escalation"] is False

def test_table_route():
    query = "What is the total cost of all active devices in Engineering?"
    state = athena_graph.invoke({"query": query})
    
    assert state["intent"] == "table_question"
    assert "answer" in state and state["answer"] is not None
    assert "12150" in state["answer"]
    assert state["needs_escalation"] is False

def test_ticket_route():
    query = "My computer won't turn on and I have a client meeting in 5 minutes! Help!"
    state = athena_graph.invoke({"query": query})
    
    assert state["intent"] == "ticket_request"
    assert "answer" in state and state["answer"] is not None
    assert "urgent" in state["answer"].lower() or "high" in state["answer"].lower()
    assert "routing" in state["answer"].lower()
    assert state["needs_escalation"] is False

def test_vague_unanswerable_escalation():
    # Deliberately vague/nonsense question
    query = "asdfghjkl qwertyuiop xyz"
    state = athena_graph.invoke({"query": query})
    
    # Confidence should be low and needs_escalation should be True
    assert state["needs_escalation"] is True
