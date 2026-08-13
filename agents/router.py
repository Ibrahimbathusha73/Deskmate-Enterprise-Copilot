from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENTS = ["docs_question", "table_question", "image_question", "ticket_request", "general_tool_use"]

def classify_intent(query: str) -> str:
    prompt = f"""Classify the user's query into exactly one label from the following list: {INTENTS}.

Use the following guidelines to select the best label:
- docs_question: Information questions about documentation, guides, code repository setup, contribution guidelines, policies, and manual text.
- table_question: Queries asking for calculations, statistics, lists, costs, or aggregations from tabular databases or spreadsheets.
- image_question: Queries mentioning pictures, images, diagrams, receipts, scans, or screenshots.
- ticket_request: Requests to triage, report, categorize, or file support/IT/HR tickets or bug reports.
- general_tool_use: Queries requesting to trigger external actions or query external APIs (e.g. fetching GitHub issues directly).

Do not include any explanation, markdown, code blocks, or extra text in your response. Output only the label itself.

Query: "{query}"
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    label = resp.choices[0].message.content.strip().strip("'\"` ")
    # Fallback to docs_question if classification matches none
    if label not in INTENTS:
        # Check if the label is contained in INTENTS
        for intent in INTENTS:
            if intent in label:
                return intent
        return "docs_question"
    return label

if __name__ == "__main__":
    test_queries = [
        "How do I contribute to this repo?",
        "What is the total cost of all active devices in engineering?",
        "Please triage this bug ticket",
        "Here is an image of the receipt"
    ]
    for q in test_queries:
        print(f"Query: {q} -> Classified Intent: {classify_intent(q)}")
