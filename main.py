from Nodes.guardrail_node import guardrail_node
from Edges.guardrail_edge import guardrail_edge
from Edges.route_transaction_tools import route_transaction_tools
from Nodes.recall_node import recall_node
from Nodes.remember_node import remember_node
from Nodes.account_agent import get_account_agent_nodes
from Nodes.knowledge_agent import get_knowledge_agent_nodes
from Nodes.transaction_agent import get_transaction_agent_nodes
from Edges.distribute_tasks_edge import distribute_tasks
from Edges.primary_classifier_route_edge import route_triage
from Nodes.aggregator_node import aggregator_node
from Nodes.orchestrator import orchestrator_node
from Nodes.primary_classifier_node import triage_router
from Nodes.worker import worker_node_function
from State.banking_state import BankingState
from langgraph.graph import StateGraph, START, END


def build_graph(all_mcp_tools, checkpointer=None, ltm_store=None):
    """
    Assembles and compiles the map-reduce banking agent.
    Pass your PostgreSQL or MemorySaver checkpointer here.
    """
    account_agent, account_tools, account_tools_condition = get_account_agent_nodes(all_mcp_tools=all_mcp_tools)
    transaction_agent, safe_tools_node, sensitive_tools_node = get_transaction_agent_nodes(all_mcp_tools)
    knowledge_agent, knowledge_tools_node, knowledge_tools_condition = get_knowledge_agent_nodes(all_mcp_tools)

    workflow = StateGraph(BankingState)

    # 1. Add Nodes
    workflow.add_node("recall_node", recall_node)
    workflow.add_node("guardrail_node", guardrail_node)
    workflow.add_node("triage_router", triage_router)
    workflow.add_node("orchestrator", orchestrator_node)

    workflow.add_node("account_agent", account_agent)
    workflow.add_node("account_tools", account_tools)

    workflow.add_node("transaction_agent", transaction_agent)
    workflow.add_node("safe_tools_node", safe_tools_node)
    workflow.add_node("sensitive_tools_node", sensitive_tools_node)

    workflow.add_node("knowledge_agent", knowledge_agent)
    workflow.add_node("knowledge_tools", knowledge_tools_node)

    workflow.add_node("worker_node", worker_node_function)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("remember_node", remember_node)
    
    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "recall_node")
    workflow.add_edge("recall_node", "guardrail_node")

    workflow.add_conditional_edges(
        "guardrail_node",
        guardrail_edge,
        {
            "triage_router": "triage_router",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "triage_router",
        route_triage,
        {
            "orchestrator": "orchestrator",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "orchestrator", 
        distribute_tasks,
        {
            "account_agent": "account_agent",
            "transaction_agent": "transaction_agent",
            "knowledge_agent": "knowledge_agent",
            "worker_node": "worker_node"
        }
    )

    workflow.add_conditional_edges(
        "account_agent",
        account_tools_condition,
        {
            "tools": "account_tools",
            END: "aggregator" 
        }
    )

    workflow.add_conditional_edges(
        "transaction_agent",
        route_transaction_tools,
        {
            "safe_tools": "safe_tools_node",
            "sensitive_tools": "sensitive_tools_node",
            "aggregator": "aggregator"
        }
    )

    workflow.add_conditional_edges(
        "knowledge_agent",
        knowledge_tools_condition,
        {
            "tools": "knowledge_tools",
            END: "aggregator" 
        }
    )
    
    # ReAct Loops
    workflow.add_edge("account_tools", "account_agent")
    workflow.add_edge("safe_tools_node", "transaction_agent")
    workflow.add_edge("sensitive_tools_node", "transaction_agent")
    workflow.add_edge("knowledge_tools", "knowledge_agent")
    
    workflow.add_edge("worker_node", "aggregator")
    workflow.add_edge("aggregator", "remember_node")
    workflow.add_edge("remember_node", END)
    
    # 3. Compilation with Human-in-the-Loop Interrupt
    
    app = workflow.compile(
        checkpointer=checkpointer,
        store=ltm_store,  # <--- Pass your SqliteKeyValueStore instance here!
        interrupt_before=["sensitive_tools_node"]
    )
    
    return app
