# Edges/route_orchestration.py
from langgraph.constants import Send
from State.banking_state import BankingState

def orchestrator_router(state: BankingState):
    """
    If complete, route to aggregator.
    If tasks exist, map them to parallel workers using the Send API.
    """
    if state.get("is_workflow_complete", False):
        return "aggregator"
        
    tasks = state.get("tasks", [])
    
    # 🌟 Spawns multiple 'worker_subgraph' nodes concurrently!
    # It passes a specialized WorkerState to each one.
    return [Send("worker_subgraph", {"task": task}) for task in tasks]