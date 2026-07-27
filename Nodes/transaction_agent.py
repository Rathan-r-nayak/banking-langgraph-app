from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import END

from State.banking_state import BankingState

def get_transaction_agent_nodes(all_mcp_tools: list, model_name: str = "gpt-4o"):
    
    # 1. Split the tools by security clearance
    safe_tools = [t for t in all_mcp_tools if t.name == "verify_account"]
    sensitive_tools = [t for t in all_mcp_tools if t.name == "execute_transfer"]
    
    # 2. Bind ALL tools to the LLM so it knows they exist
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(safe_tools + sensitive_tools)
    
    def transaction_agent_node(state: BankingState):
        # (Inject system prompt here)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # 3. Create TWO separate ToolNodes
    safe_tools_node = ToolNode(safe_tools)
    sensitive_tools_node = ToolNode(sensitive_tools)
    
    return transaction_agent_node, safe_tools_node, sensitive_tools_node