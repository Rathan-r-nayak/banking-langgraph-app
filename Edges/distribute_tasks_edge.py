def distribute_tasks(state):
    tasks = state.get("tasks", [])
    if not tasks:
        return "aggregator" # Or END
    
    # Grab the current task
    task = tasks[0]
    
    # Check for 'type' first (since your Task schema uses 'type'), then 'agent'
    if isinstance(task, dict):
        assigned_agent = task.get("type") or task.get("agent", "worker_node")
    else:
        assigned_agent = getattr(task, "type", None) or getattr(task, "agent", "worker_node")
    
    # Map the agent name directly to your graph node names
    if assigned_agent == "account_agent":
        return "account_agent"
    elif assigned_agent == "transaction_agent":
        return "transaction_agent"
    elif assigned_agent == "knowledge_agent":
        return "knowledge_agent"
    else:
        # Fallback if the name doesn't match
        return "worker_node"