import torch
torch.set_num_threads(1)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import streamlit as st

@st.cache_resource
def get_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="data/chroma_db")

@st.cache_resource
def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection("deskmate_docs")

def __getattr__(name):
    if name == "collection":
        return get_collection()
    if name == "embed_model":
        return get_embed_model()
    if name == "reranker":
        return get_reranker()
    if name == "client":
        return get_chroma_client()
    raise AttributeError(f"module {__name__} has no attribute {name}")

def hybrid_search(query, top_k=20, final_k=3):
    embed_model = get_embed_model()
    reranker = get_reranker()
    collection = get_collection()

    # dense search
    q_emb = embed_model.encode([query]).tolist()
    dense_results = collection.query(query_embeddings=q_emb, n_results=top_k)
    
    if not dense_results or not dense_results.get("documents") or not dense_results["documents"][0]:
        return []
        
    docs = dense_results["documents"][0]
    ids = dense_results["ids"][0]

    # sparse (BM25) over the same candidate pool for fusion
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    # Sort the dense candidates by their BM25 scores
    fused = sorted(zip(ids, docs, bm25_scores), key=lambda x: -x[2])[:top_k]

    if not fused:
        return []

    # cross-encoder rerank on fused candidates
    pairs = [[query, d] for _, d, _ in fused]
    rerank_scores = reranker.predict(pairs)
    
    # Predict can return a single float if there's only one pair, but predict usually returns an array/list.
    # To handle single element predictions correctly, make sure we cast or handle correctly.
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
        
    reranked = sorted(zip(fused, rerank_scores), key=lambda x: -x[1])[:final_k]

    return [{"id": r[0][0], "text": r[0][1], "score": float(r[1])} for r in reranked]

if __name__ == "__main__":
    # Small test query when run directly
    import sys
    q = "how to contribute to transformers"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    print(f"Query: {q}")
    res = hybrid_search(q)
    for r in res:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}\nText: {r['text'][:200]}...\n")

