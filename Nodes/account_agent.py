from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from State.banking_state import BankingState
from Utils.Logger import get_logger
from dotenv import load_dotenv

logger = get_logger("ACCOUNT_AGENT")

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

def get_account_agent_nodes(all_mcp_tools: list[Any]):
    """
    Returns the reasoning node, the tool execution node, and the routing condition.
    """
    
    ACCOUNT_TOOL_NAMES = {
        "create_new_account",
        "check_balance",
        "get_account",
        "get_all_accounts",
        "deposit_money",
        "withdraw_money"
    }
    
    account_tools = [t for t in all_mcp_tools if t.name in ACCOUNT_TOOL_NAMES]
    logger.info(f"💳 Account Agent initialized with {len(account_tools)} tools.")
    
    # 3. Define the Reasoning Node (Now ASYNC)
    async def account_agent_node(state: BankingState):
        logger.info("--- 💳 RUNNING ACCOUNT AGENT ---")
        
        # 🌟 THE FIX: Instantiate the LLM INSIDE the async node!
        # This guarantees the async network client binds to the active event loop.
        llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash', temperature=0)
        llm_with_tools = llm.bind_tools(account_tools)
        
        messages = state.get("messages", [])
        
        # Inject the system prompt if it's not already at the front
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + messages
            
        try:
            response = await llm_with_tools.ainvoke(messages)
            updates = {"messages": [response]}
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(f"🛠️ Account Agent requested tool call(s): {tool_names}")
            elif response.content:
                snippet = response.content[:100].replace('\n', ' ')
                logger.info(f"💬 Account Agent generated final response: '{snippet}...'")
                
                existing_responses = state.get("worker_responses", [])
                updates["worker_responses"] = existing_responses + [response.content]
                logger.info("✅ Pushed Account Agent final response to worker_responses.")
                
            return updates
            
        except Exception as e:
            logger.error(f"❌ Account Agent execution failed: {e}")
            raise e

    # 4. Define the Execution Node
    account_tools_node = ToolNode(account_tools)
    
    return account_agent_node, account_tools_node, tools_condition