from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
PRIORITIES = ["low", "medium", "high", "urgent"]

def ticket_agent(query: str) -> dict:
    prompt = f"""Classify the priority of this support request into exactly one of {PRIORITIES}, and give a one-sentence routing recommendation.
Request: "{query}"
Respond ONLY with a JSON object matching this schema:
{{
  "priority": "one of low, medium, high, urgent",
  "routing": "one-sentence routing recommendation"
}}
Do not include any explanation, markdown, code blocks, or text outside the JSON.
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    
    content = resp.choices[0].message.content.strip()
    
    # Clean up code blocks if they exist
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
        
    try:
        data = json.loads(content)
        priority = data.get("priority", "medium").strip().lower()
        if priority not in PRIORITIES:
            priority = "medium"
        return {
            "priority": priority,
            "routing": data.get("routing", "Route to support team."),
            "raw": content
        }
    except Exception:
        # Fallback parsing
        # Try to search for keywords in case of malformed JSON
        priority = "medium"
        for p in PRIORITIES:
            if p in content.lower():
                priority = p
                break
        return {
            "priority": priority,
            "routing": "Could not parse JSON. Route to IT support.",
            "raw": content
        }

if __name__ == "__main__":
    queries = [
        "My laptop screen is completely black and won't turn on. I have a client presentation in 10 minutes!",
        "Need help configuring my email signature",
    ]
    for q in queries:
        print(f"Request: {q}\nResult: {ticket_agent(q)}\n")
