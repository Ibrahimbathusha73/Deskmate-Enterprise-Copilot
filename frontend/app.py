import streamlit as st
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

st.set_page_config(
    page_title="ATHENA",
    layout="centered"
)

@st.cache_resource
def ensure_index_built():
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection("athena_docs")
    if collection.count() == 0:
        # index is empty — run the same ingestion Phase 1 used
        from ingestion.fetch_github_issues import fetch_issues
        from ingestion.chunk_and_embed import process_github_issues
        
        try:
            fetch_issues()
        except Exception as e:
            st.warning(f"Note: Could not fetch fresh issues from GitHub ({e}). Using pre-packaged issue database.")
            
        if not os.path.exists("data/tickets/github_issues.json"):
            st.error("Error: GitHub issues database not found. Cannot initialize vector index.")
            return False
            
        process_github_issues("data/tickets/github_issues.json")
    return True

# Initialize vector database index if empty
with st.spinner("Building knowledge base..."):
    ensure_index_built()

from orchestrator.graph import athena_graph

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "escalated_count" not in st.session_state:
    st.session_state.escalated_count = 0
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
if "temp_query" not in st.session_state:
    st.session_state.temp_query = ""

# Inject global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* Global styling overrides */
.stApp {
    background-color: #14171C !important;
    color: #E8EAED !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #1B1F26 !important;
    border-right: 0.5px solid #2A2F38 !important;
}

/* Hide default streamlit header elements */
[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Sidebar Custom Content styling */
.sidebar-container {
    padding: 10px 5px;
}

.wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: 0.05em;
    color: #E8EAED;
    margin-bottom: 2px;
}

.wordmark-sub {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    color: #8B93A1;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 20px;
}

.stat-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #8B93A1;
    margin-bottom: 8px;
}

.stat-val-teal {
    color: #3FA7A0;
    font-weight: 700;
}

.stat-val-amber {
    color: #D9A441;
    font-weight: 700;
}

.stat-val-normal {
    color: #E8EAED;
    font-weight: 700;
}

.sidebar-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: #8B93A1;
    line-height: 1.5;
}

/* Chat area styling */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 30px;
    width: 100%;
}

.user-bubble-container {
    display: flex;
    justify-content: flex-end;
    width: 100%;
    margin-top: 10px;
    margin-bottom: 10px;
}

.user-bubble {
    background-color: #1B1F26;
    border: 0.5px solid #2A2F38;
    border-radius: 8px;
    padding: 10px 14px;
    max-width: 85%;
    color: #E8EAED;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14.5px;
    line-height: 1.5;
}

.assistant-response {
    width: 100%;
    color: #E8EAED;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14.5px;
    line-height: 1.5;
    margin-top: 15px;
    margin-bottom: 5px;
    text-align: left;
}

.manifest-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background-color: #1B1F26;
    padding: 10px 14px;
    margin-top: 8px;
    margin-bottom: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
}

.manifest-strip-teal {
    border-left: 2px solid #3FA7A0 !important;
    border-radius: 0 !important;
}

.manifest-strip-amber {
    border-left: 2px solid #D9A441 !important;
    border-radius: 0 !important;
}

.manifest-label {
    color: #8B93A1;
}

.manifest-val {
    color: #E8EAED;
    font-weight: 500;
}

.sources-box {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: #8B93A1;
    background-color: #14171C;
    border: 0.5px solid #2A2F38;
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 5px;
    margin-bottom: 20px;
    width: 100%;
}

.sources-title {
    font-size: 10px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #8B93A1;
    margin-bottom: 6px;
    font-weight: 700;
}

.source-item {
    margin-bottom: 4px;
    word-break: break-all;
}

.source-link {
    color: #3FA7A0;
    text-decoration: none;
}
.source-link:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Render Sidebar Content
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-container">
        <div class="wordmark">ATHENA</div>
        <div class="wordmark-sub">Enterprise Secure Copilot</div>
        <hr style="margin: 15px 0; border: 0; border-top: 0.5px solid #2A2F38;" />
        <div class="stat-item">Total Queries: <span class="stat-val-normal">{st.session_state.total_queries}</span></div>
        <div class="stat-item">Escalated: <span class="stat-val-amber">{st.session_state.escalated_count}</span></div>
        <div class="stat-item">Cache Hits: <span class="stat-val-teal">{st.session_state.cache_hits}</span></div>
        <hr style="margin: 15px 0; border: 0; border-top: 0.5px solid #2A2F38;" />
        <div class="sidebar-desc">
            Athena is a multi-agent secure copilot. Triages queries, executes database lookup, and references documentation with automatic caching and escalation policies.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render Main Conversation Panel
if not st.session_state.history:
    st.markdown("""
    <div style="margin-top: 120px; text-align: center; font-family: 'IBM Plex Sans', sans-serif; color: #8B93A1; font-size: 15px;">
        Awaiting queries...
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-bubble-container">
                <div class="user-bubble">{message["text"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Render response
            st.markdown(f'<div class="assistant-response">{message["text"]}</div>', unsafe_allow_html=True)
            
            # Manifest Strip styling
            is_escalated = message["needs_escalation"]
            border_class = "manifest-strip-amber" if is_escalated else "manifest-strip-teal"
            
            st.markdown(f"""
            <div class="manifest-strip {border_class}">
                <div><span class="manifest-label">AGENT:</span> <span class="manifest-val">{message["agent"]}</span></div>
                <div><span class="manifest-label">MODEL:</span> <span class="manifest-val">{message["model"]}</span></div>
                <div><span class="manifest-label">CONFIDENCE:</span> <span class="manifest-val">{message["confidence"]:.2f}</span></div>
                <div><span class="manifest-label">LATENCY:</span> <span class="manifest-val">{message["latency"]:.3f}S</span></div>
                <div><span class="manifest-label">CACHE:</span> <span class="manifest-val">{message["cache_status"]}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources Details dropdown
            if not is_escalated and message.get("sources"):
                sources_list_html = ""
                for idx, c in enumerate(message["sources"]):
                    src_link = c.get("source") or "#"
                    sources_list_html += f'<div class="source-item">{idx+1}. <a class="source-link" href="{src_link}" target="_blank">{c.get("id")}</a></div>'
                
                st.markdown(f"""
                <details class="sources-box">
                    <summary style="cursor: pointer; outline: none; list-style: none;" class="sources-title">SOURCES</summary>
                    <div style="margin-top: 8px;">
                        {sources_list_html}
                    </div>
                </details>
                """, unsafe_allow_html=True)

# Main Query input container
query = st.text_input("Ask a question", placeholder="Type your query here...", key="user_query_val")

# Process query input
if query and query != st.session_state.temp_query:
    st.session_state.temp_query = query
    
    # Process
    import time
    start_time = time.time()
    try:
        result = athena_graph.invoke({"query": query})
        latency = time.time() - start_time
        
        answer = result.get("answer", "No response generated.")
        intent = result.get("intent")
        confidence = result.get("confidence", 0.0)
        needs_escalation = result.get("needs_escalation", False)
        agent = result.get("agent", "UNKNOWN_AGENT")
        model = result.get("model", "N/A")
        cache_status = result.get("cache_status", "MISS")
        sources = result.get("retrieved_chunks") or []
        
        # Update stats
        st.session_state.total_queries += 1
        if needs_escalation:
            st.session_state.escalated_count += 1
        if cache_status == "HIT":
            st.session_state.cache_hits += 1
            
        # Add to history
        st.session_state.history.append({"role": "user", "text": query})
        st.session_state.history.append({
            "role": "assistant",
            "text": answer,
            "agent": agent,
            "model": model,
            "confidence": confidence,
            "latency": latency,
            "cache_status": cache_status,
            "sources": sources,
            "needs_escalation": needs_escalation
        })
    except Exception as e:
        st.session_state.history.append({"role": "user", "text": query})
        st.session_state.history.append({
            "role": "assistant",
            "text": f"Error executing graph: {e}",
            "agent": "ORCHESTRATOR",
            "model": "N/A",
            "confidence": 0.0,
            "latency": 0.0,
            "cache_status": "N/A",
            "sources": [],
            "needs_escalation": True
        })
    
    st.session_state.user_query_val = ""
    st.rerun()
