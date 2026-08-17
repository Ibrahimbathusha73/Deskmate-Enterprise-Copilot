import streamlit as st
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

from orchestrator.graph import athena_graph

st.set_page_config(
    page_title="Athena — Enterprise Copilot",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Athena — Enterprise Copilot")
st.write("Welcome to Athena, your secure multi-agent RAG copilot. Ask any question about repository docs, ticketing systems, or asset costs.")

query = st.text_input("Ask a question", placeholder="Type your query here...")

if query:
    with st.spinner("Invoking agents and analyzing query..."):
        try:
            result = athena_graph.invoke({"query": query})
            
            st.subheader("Response")
            st.write(result.get("answer", "No response generated."))
            
            if result.get("needs_escalation"):
                st.warning("⚠️ Low confidence warning: This interaction has been logged to the SQLite escalation database and routed to a human engineer.")
            
            with st.expander("Sources & Retrieval Metadata"):
                intent = result.get("intent")
                st.write(f"**Classified Intent:** `{intent}`")
                
                chunks = result.get("retrieved_chunks") or []
                if chunks:
                    for i, c in enumerate(chunks):
                        st.markdown(f"**Source {i+1}:** `{c.get('id')}` | **Score:** `{c.get('score', 0.0):.4f}`")
                        st.caption(c.get("text", ""))
                        st.divider()
                else:
                    st.info("No external document sources were retrieved for this query intent.")
        except Exception as e:
            st.error(f"An error occurred while executing the graph: {e}")
