import operator
from typing import Annotated, Any, Literal, Sequence, TypedDict
from langgraph.store.memory import InMemoryStore
from langchain_classic.schema import BaseMessage
from langgraph.graph import add_messages

def merge_lists(left: list | None, right: list | None) -> list:
    return (left or []) + (right or [])

class Task(TypedDict):
    description: str
    type: Literal["account_agent", "transaction_agent", "knowledge_agent"]

class BankingState(TypedDict, total=False):
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memories: str
    
    is_safe: bool
    requires_workflow: bool
    documents: list[dict]
    is_sufficient: bool
    tasks: list[Task]
    worker_responses: Annotated[list[str], merge_lists]
    generation: str

class WorkerState(TypedDict):
    task: Task # Or use your AnnotatedTask Pydantic model
    worker_responses:list
    
    
memory_store = InMemoryStore()