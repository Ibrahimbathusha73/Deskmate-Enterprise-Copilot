from sentence_transformers import SentenceTransformer
import chromadb
import json
import os

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection("deskmate_docs")

def chunk_text(text, size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

from ops.pii_redact import redact

def embed_and_store(docs: list[dict]):
    """docs = [{'id': str, 'text': str, 'source': str}]"""
    for doc in docs:
        redacted_text = redact(doc["text"])
        chunks = chunk_text(redacted_text)
        if not chunks:
            continue
        embeddings = model.encode(chunks).tolist()
        collection.add(
            ids=[f"{doc['id']}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": doc["source"]} for _ in chunks],
        )

def process_github_issues(json_path="data/tickets/github_issues.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist. Please run fetch_github_issues.py first.")
        
    with open(json_path, "r") as f:
        issues = json.load(f)
        
    formatted_docs = []
    for issue in issues:
        # Note: Github pull requests are also returned by the issues API,
        # but they are fine as a text source.
        number = issue.get("number")
        title = issue.get("title", "")
        body = issue.get("body") or ""
        state = issue.get("state", "")
        html_url = issue.get("html_url", "")
        
        text_content = f"Issue #{number}: {title}\nState: {state}\n\n{body}"
        formatted_docs.append({
            "id": f"github_issue_{number}",
            "text": text_content,
            "source": html_url
        })
        
    print(f"Loaded {len(formatted_docs)} issues. Chunking, embedding, and storing in ChromaDB...")
    embed_and_store(formatted_docs)
    print(f"Ingested documents. Total count in ChromaDB collection: {collection.count()}")

if __name__ == "__main__":
    process_github_issues()
