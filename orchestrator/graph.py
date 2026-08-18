from langgraph.graph import StateGraph, END
from orchestrator.state import DeskmateState
from agents.router import classify_intent
from agents.docs_rag_agent import docs_rag_agent
from agents.table_agent import table_agent
from agents.ticket_agent import ticket_agent
from agents.vision_agent import vision_agent

from ops.pii_redact import redact

def route_node(state: DeskmateState) -> DeskmateState:
    state["intent"] = classify_intent(state["query"])
    return state

def docs_node(state: DeskmateState) -> DeskmateState:
    result = docs_rag_agent(state["query"])
    state["answer"] = redact(result["answer"])
    state["retrieved_chunks"] = result["chunks"]
    state["agent"] = "DOCS_RAG_AGENT"
    state["model"] = result.get("model_used", "openai/gpt-oss-20b")
    state["cache_status"] = result.get("cache_status", "MISS")
    
    # Assess confidence using the Cross-Encoder score of the top retrieved chunk.
    # Scores > -4.0 suggest a relevant document match.
    if result["chunks"]:
        max_score = max(c["score"] for c in result["chunks"])
        if max_score > -4.0:
            state["confidence"] = 0.9
        else:
            # Low relevance score triggers escalation
            state["confidence"] = 0.3
    else:
        state["confidence"] = 0.2
    return state

def table_node(state: DeskmateState) -> DeskmateState:
    import pandas as pd
    import os
    csv_path = "data/sample_table.csv"
    state["agent"] = "TABLE_AGENT"
    state["model"] = "openai/gpt-oss-20b"
    state["cache_status"] = "N/A"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        result = table_agent(state["query"], df)
        state["answer"] = redact(f"Calculation Result:\n{result['answer']}\n(Expression: {result['code']})")
        if "Could not compute:" in result["answer"]:
            state["confidence"] = 0.3
        else:
            state["confidence"] = 0.9
    else:
        state["answer"] = redact(f"Error: Database file {csv_path} not found.")
        state["confidence"] = 0.1
    return state

def ticket_node(state: DeskmateState) -> DeskmateState:
    result = ticket_agent(state["query"])
    state["answer"] = redact(f"Priority: {result['priority']}\nRouting: {result['routing']}")
    state["agent"] = "TICKET_AGENT"
    state["model"] = "openai/gpt-oss-20b"
    state["cache_status"] = "N/A"
    state["confidence"] = 0.8
    return state

def vision_node(state: DeskmateState) -> DeskmateState:
    result = vision_agent("", state["query"])
    state["answer"] = redact(result["answer"])
    state["agent"] = "VISION_AGENT"
    state["model"] = "N/A"
    state["cache_status"] = "N/A"
    # Since it is a stub / not implemented, confidence is low to force escalation
    state["confidence"] = 0.4
    return state

def escalation_check(state: DeskmateState) -> DeskmateState:
    needs_esc = (state.get("confidence") or 0) < 0.5
    state["needs_escalation"] = needs_esc
    if needs_esc:
        try:
            import sqlite3
            import datetime
            import os
            
            db_dir = "data"
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "escalations.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query TEXT,
                    agent_attempted TEXT,
                    confidence REAL,
                    reason TEXT
                )
            """)
            
            agent_attempted = state.get("intent", "unknown")
            confidence = state.get("confidence", 0.0)
            reason = f"Confidence {confidence} is below escalation threshold 0.5"
            
            cursor.execute("""
                INSERT INTO escalations (timestamp, query, agent_attempted, confidence, reason)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.datetime.utcnow().isoformat(),
                state.get("query", ""),
                agent_attempted,
                confidence,
                reason
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging escalation: {e}")
            
        state["answer"] = "I'm not confident enough in this answer — this has been routed to a human for follow-up."
    return state

def route_decision(state: DeskmateState) -> str:
    intent = state.get("intent", "docs_question")
    return {
        "docs_question": "docs",
        "table_question": "table",
        "ticket_request": "ticket",
        "image_question": "vision",
        "general_tool_use": "docs"
    }.get(intent, "docs")

# Build the LangGraph StateGraph
graph = StateGraph(DeskmateState)

# Add all nodes
graph.add_node("router", route_node)
graph.add_node("docs", docs_node)
graph.add_node("table", table_node)
graph.add_node("ticket", ticket_node)
graph.add_node("vision", vision_node)
graph.add_node("escalation_check", escalation_check)

# Define entry point and edges
graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "docs": "docs",
        "table": "table",
        "ticket": "ticket",
        "vision": "vision"
    }
)
graph.add_edge("docs", "escalation_check")
graph.add_edge("table", "escalation_check")
graph.add_edge("ticket", "escalation_check")
graph.add_edge("vision", "escalation_check")
graph.add_edge("escalation_check", END)

deskmate_graph = graph.compile()

if __name__ == "__main__":
    import pprint
    # Visual check of graph compilation
    print("Graph compiled successfully. Executing test query:")
    result = deskmate_graph.invoke({"query": "How do I contribute to this repo?"})
    pprint.pprint(result)
