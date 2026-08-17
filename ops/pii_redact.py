import re

PATTERNS = {
    "email": r"[\w\.-]+@[\w\.-]+\.\w+",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

def redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    for label, pattern in PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{label.upper()}]", text)
    return text
