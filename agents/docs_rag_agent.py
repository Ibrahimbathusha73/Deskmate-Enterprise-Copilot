from retrieval.hybrid_search import hybrid_search
from groq import Groq
import os
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context
from ops.semantic_cache import get_cached, set_cache

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def is_unanswerable(answer: str) -> bool:
    unanswerable_keywords = [
        "does not contain",
        "not enough information",
        "insufficient info",
        "cannot answer",
        "not provide",
        "not mentioned",
        "no information",
        "unable to find",
        "context does not",
        "i do not have",
        "i don't have"
    ]
    ans_lower = answer.lower()
    return any(keyword in ans_lower for keyword in unanswerable_keywords)

@observe(as_type="generation", name="docs_rag_agent")
def docs_rag_agent(query: str) -> dict:
    # 1. Check Redis semantic cache
    cache_res = get_cached(query)
    if cache_res:
        cached_answer, chunks = cache_res
        langfuse_context.update_current_observation(
            input=query,
            output=cached_answer,
            model="semantic-cache-hit",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        )
        return {
            "answer": cached_answer,
            "chunks": chunks
        }

    # 2. Run retrieval
    chunks = hybrid_search(query)
    if not chunks:
        answer = "I could not find any relevant documentation in my knowledge base to answer this question."
        langfuse_context.update_current_observation(
            input=query,
            output=answer,
            model="none",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        )
        return {
            "answer": answer,
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

    # 3. First tier call: llama-3.1-8b-instant (low cost)
    print("[MODEL TIERING] Querying llama-3.1-8b-instant...")
    resp_8b = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer_8b = resp_8b.choices[0].message.content.strip()
    prompt_tokens = resp_8b.usage.prompt_tokens
    completion_tokens = resp_8b.usage.completion_tokens
    model_used = "llama-3.1-8b-instant"
    final_answer = answer_8b

    # 4. Check for unanswerability, fallback to llama-3.3-70b-versatile (high performance)
    if is_unanswerable(answer_8b):
        print("[MODEL TIERING] 8B returned unanswerable response. Falling back to llama-3.3-70b-versatile...")
        resp_70b = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer_70b = resp_70b.choices[0].message.content.strip()
        prompt_tokens += resp_70b.usage.prompt_tokens
        completion_tokens += resp_70b.usage.completion_tokens
        model_used = "llama-3.3-70b-versatile"
        final_answer = answer_70b

    # 5. Save to semantic cache
    set_cache(query, final_answer, chunks)

    # 6. Log metrics in Langfuse
    langfuse_context.update_current_observation(
        input=prompt,
        output=final_answer,
        model=model_used,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    )

    return {
        "answer": final_answer,
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
