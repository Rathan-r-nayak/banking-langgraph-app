# Graph/rag_sub_graph.py
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from Nodes.web_search_node import web_search_node
from Nodes.rag_nodes.retrive_documents_node import retrieve_node
from Nodes.rag_nodes.evaluate_node import evaluate_node
from Nodes.rag_nodes.rewrite_query_node import rewrite_node
from Nodes.rag_nodes.generate_node import generate_node
from State.rag_state import RagState
from Utils.Logger import get_logger
from Utils.web_search_helpers import ask_web_search, check_relevance, check_web_search_approval

logger = get_logger("RAG_SUBGRAPH")


def get_rag_subgraph():
    builder = StateGraph(RagState)
    
    # Add Core Nodes
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("rewrite_node", rewrite_node)
    builder.add_node("generate_node", generate_node)
    
    # Add Web Search Nodes
    builder.add_node("ask_web_search", ask_web_search)
    builder.add_node("web_search_node", web_search_node)
    
    # Connect standard RAG loop
    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "evaluate_node")
    
    builder.add_conditional_edges(
        "evaluate_node",
        check_relevance,
        {
            "generate_node": "generate_node",
            "rewrite_node": "rewrite_node",
            "ask_web_search": "ask_web_search" # Route to the HITL node
        }
    )
    builder.add_edge("rewrite_node", "retrieve_node")
    
    # Connect Web Search Loop
    builder.add_conditional_edges(
        "ask_web_search", 
        check_web_search_approval, 
        {
            "web_search_node": "web_search_node", 
            "generate_node": "generate_node"
        }
    )
    builder.add_edge("web_search_node", "generate_node")
    builder.add_edge("generate_node", END)
    
    # 🌟 Compile with an interrupt AFTER asking the user
    return builder.compile(interrupt_after=["ask_web_search"])