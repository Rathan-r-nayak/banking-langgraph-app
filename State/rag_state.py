# State/rag_state.py
from typing import TypedDict
from langchain_core.messages import BaseMessage

class RagState(TypedDict):
    question: str
    messages: list[BaseMessage]
    documents: str
    relevance_score: str
    knowledge_retries: int
    generation: str