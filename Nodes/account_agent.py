from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from State.banking_state import BankingState


ACCOUNT_SYSTEM_PROMPT = """You are Secure Bank's Account Specialist.
Your job is to assist customers with account management tasks ONLY.

Available capabilities:
- Checking account balances
- Opening new savings or checking accounts
- Fetching account details
- Processing deposits and withdrawals

Rules:
1. Always maintain a polite, clear, and secure tone.
2. If an operation succeeds or fails, explain the outcome concisely.
3. If the user asks to transfer money to another person, you MUST politely inform them that you cannot do that and ask them to repeat the request so the Orchestrator can route them to the Transaction Agent."""



def get_account_agent_nodes(all_mcp_tools: list[Any], model_name: str = "gpt-4o-mini"):
    """
    Returns the reasoning node, the tool execution node, and the routing condition.
    """
    
    # 1. Filter tools to strictly isolate domain capabilities
    ACCOUNT_TOOL_NAMES = {
        "create_new_account",
        "check_balance",
        "get_account",
        "get_all_accounts",
        "deposit_money",
        "withdraw_money"
    }
    
    account_tools = [t for t in all_mcp_tools if t.name in ACCOUNT_TOOL_NAMES]
    
    # 2. Initialize LLM and bind the filtered tools
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(account_tools)
    
    # 3. Define the Reasoning Node
    def account_agent_node(state: BankingState):
        messages = state["messages"]
        
        # Inject the system prompt if it's not already at the front
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + messages
            
        # The LLM reads the history and decides to either reply with text OR output a tool_call
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 4. Define the Execution Node
    # LangGraph's ToolNode automatically executes the requested FastMCP tool
    account_tools_node = ToolNode(account_tools)
    
    return account_agent_node, account_tools_node, tools_condition