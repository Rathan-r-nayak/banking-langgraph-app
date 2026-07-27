from State.banking_state import BankingState


def route_transaction_tools(state: BankingState):
    """Routes tool calls to either the safe zone or the sensitive (paused) zone."""
    last_message = state["messages"][-1]
    
    # If the LLM didn't call a tool, it's done. Route to aggregator.
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "aggregator"
        
    # Check WHICH tool the LLM is trying to call
    tool_name = last_message.tool_calls[0]["name"]
    
    if tool_name == "execute_transfer":
        return "sensitive_tools" # This path will be paused!
    else:
        return "safe_tools"      # This path runs instantly