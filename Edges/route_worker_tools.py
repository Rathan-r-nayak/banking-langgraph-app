# Edges/route_worker_tools.py
from State.banking_state import WorkerState
from langgraph.graph import END

def route_worker_tools(state: WorkerState):
    """Routes tools inside the Sub-Graph."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # 🌟 If no tools were called, the task is finished! 
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
        
    SENSITIVE_TOOLS = {"transfer_money", "pay_bill", "update_password"}
    
    for tc in last_message.tool_calls:
        if tc["name"] == "search_bank_policies":
            return "rag_subgraph"
        if tc["name"] in SENSITIVE_TOOLS:
            return "sensitive_tools_node"
            
    return "safe_tools_node"