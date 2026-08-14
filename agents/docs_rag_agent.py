from retrieval.hybrid_search import hybrid_search
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def docs_rag_agent(query: str) -> dict:
    chunks = hybrid_search(query)
    
    if not chunks:
        return {
            "answer": "I could not find any relevant documentation in my knowledge base to answer this question.",
            "chunks": []
        }
        
    context_parts = []
    for c in chunks:
        context_parts.append(f"Source ID: {c['id']}\nContent: {c['text']}")
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below. 
Cite your sources by mentioning their source ID. If the context does not contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {query}
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "chunks": chunks
    }

if __name__ == "__main__":
    q = "what is the pipeline device validation enhancement?"
    print(f"Query: {q}")
    res = docs_rag_agent(q)
    print("Answer:\n", res["answer"])
    print("\nSource Chunks:")
    for c in res["chunks"]:
        print(f"- {c['id']} (Score: {c['score']:.4f})")
