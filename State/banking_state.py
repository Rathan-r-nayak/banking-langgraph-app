import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langgraph.graph import add_messages
from langgraph.store.memory import InMemoryStore
from langchain_classic.schema import BaseMessage


def merge_lists(left: list | None, right: list | None) -> list:
    if right == []:
        return []

    return (left or []) + (right or [])


class Task(TypedDict):
    description: str
    type: Literal[
        "account_agent",
        "transaction_agent",
        "knowledge_agent",
    ]


class BankingState(TypedDict, total=False):
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memories: str

    is_safe: bool
    requires_workflow: bool
    documents: list[dict]

    tasks: list[Task]
    worker_responses: Annotated[list[str], merge_lists]

    is_sufficient: bool
    relevance_score: str
    knowledge_retries: int

    generation: str


class WorkerState(TypedDict):
    task: Task
    worker_responses: list


memory_store = InMemoryStore()