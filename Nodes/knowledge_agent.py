from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from State.banking_state import BankingState

KNOWLEDGE_SYSTEM_PROMPT = """You are Secure Bank's Knowledge & Policy Specialist.
Your role is to answer complex queries regarding bank policies, corporate guidelines, loan eligibility, and government incentive schemes (e.g., PMVBRY).

You have access to a sophisticated GraphRAG knowledge base.

Rules:
1. NEVER guess or hallucinate bank policies. You must ALWAYS use your GraphRAG search tools to retrieve accurate information before answering.
2. GraphRAG returns interconnected entities and documents. Synthesize these multiple sources into a clear, unified answer for the user.
3. If the retrieved context is insufficient, contradictory, or empty, explicitly state what information is missing and ask the user to clarify. Do not attempt to fill in the blanks yourself.
4. Keep your final explanations structured and easy to read."""

def get_knowledge_agent_nodes(all_mcp_tools: list[Any], model_name: str = "gpt-4o-mini"):
    """
    Returns the reasoning node, the tool execution node, and the routing condition.
    """
    
    # 1. Filter tools to isolate GraphRAG capabilities
    # Update these names to match whatever you named your FastMCP GraphRAG tools
    KNOWLEDGE_TOOL_NAMES = {
        "query_graphrag", 
        "search_policies",
        "get_entity_relationships"
    }
    
    knowledge_tools = [t for t in all_mcp_tools if t.name in KNOWLEDGE_TOOL_NAMES]
    
    # 2. Initialize LLM and bind the filtered tools
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(knowledge_tools)
    
    # 3. Define the Reasoning Node
    def knowledge_agent_node(state: BankingState):
        messages = state.get("messages", [])
        
        # Inject the system prompt if it's not already at the front
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=KNOWLEDGE_SYSTEM_PROMPT)] + messages
            
        # The LLM reads the history and decides to either reply with text OR output a tool_call
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 4. Define the Execution Node
    knowledge_tools_node = ToolNode(knowledge_tools)
    
    return knowledge_agent_node, knowledge_tools_node, tools_condition