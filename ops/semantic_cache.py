import redis
import json
import hashlib
import torch
from sentence_transformers import SentenceTransformer, util

# Ensure PyTorch uses 1 thread to avoid thread contention issues in execution environments
torch.set_num_threads(1)

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
SIM_THRESHOLD = 0.92

def get_cached(query: str):
    try:
        keys = r.keys("cache:*")
        if not keys:
            return None
            
        q_emb = model.encode(query, convert_to_tensor=True)
        for k in keys:
            val = r.get(k)
            if not val:
                continue
            entry = json.loads(val)
            if "embedding" not in entry:
                continue
            # Convert cached list back to PyTorch tensor on same device
            cached_emb = torch.tensor(entry["embedding"], device=q_emb.device)
            similarity = util.cos_sim(q_emb, cached_emb).item()
            if similarity > SIM_THRESHOLD:
                print(f"[CACHE HIT] Found match in semantic cache: '{entry['query']}' (Similarity: {similarity:.4f})")
                return entry["answer"], entry.get("chunks", [])
    except Exception as e:
        print(f"[CACHE ERROR] Failed to query cache: {e}")
    return None

def set_cache(query: str, answer: str, chunks: list):
    try:
        # Generate embedding once for storage
        q_emb = model.encode(query).tolist()
        # Create a stable key using SHA256 hash of the query
        h = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        key = f"cache:{h}"
        entry = {
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "embedding": q_emb
        }
        r.set(key, json.dumps(entry), ex=86400)
        print(f"[CACHE SET] Query cached: '{query}'")
    except Exception as e:
        print(f"[CACHE ERROR] Failed to write cache: {e}")
