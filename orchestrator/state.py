from typing import TypedDict, Optional, List

class AthenaState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    answer: Optional[str]
    confidence: Optional[float]
    needs_escalation: Optional[bool]
