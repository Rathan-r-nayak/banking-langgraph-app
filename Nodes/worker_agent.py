from typing import Any, TypedDict
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from Schema.task import Task
from Utils.Logger import get_logger
from Config.llm_config import BASE_URL, API_KEY
from Utils.helpers import get_short_term_memory

logger = get_logger("WORKER_AGENT")

# 🌟 Define the specific state this parallel node receives
class WorkerState(TypedDict):
    task: Task
    messages: list

@tool
def search_bank_policies(query: str) -> str:
    """Search Secure Bank's policy documents, loan terms, interest rates, rules, and FAQs."""
    pass

WORKER_SYSTEM_PROMPT = """You are the Banking Worker Agent.
Your ONLY job is to execute the "SYSTEM INSTRUCTION TO WORKER" provided by the Orchestrator.

1. Read the instruction.
2. Call the necessary tool(s) to fetch the data or perform the action.
3. Once the tool returns the data, briefly summarize the result so the Orchestrator can read it.
4. Do NOT address the user directly. Address the Orchestrator.
"""

def get_worker_agent_node(all_mcp_tools: list[Any]):
    all_tools = all_mcp_tools + [search_bank_policies]
    
    async def worker_agent_node(state: WorkerState):
        logger.info("--- 👷 RUNNING WORKER AGENT ---")
        
        # 🌟 Extract the isolated task passed via the Send() API
        task = state["task"]
        logger.info(f"⚡ Executing Task in Parallel: {task.task_id} - {task.description}") 

        fresh_llm = ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-4o-mini", # Worker can be the cheaper, faster model!
            api_key=API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0),
            http_async_client=httpx.AsyncClient(verify=False, timeout=120.0),
            temperature=0
        )

        llm_with_tools = fresh_llm.bind_tools(all_tools)
        
        # 🌟 Check if we are already in the middle of executing this task

        all_messages = state.get("messages", [])
    
        # 2. Trim to 6 messages
        recent_messages = get_short_term_memory(all_messages, k=6)

        final_messages = [
            SystemMessage(content=WORKER_SYSTEM_PROMPT),
            HumanMessage(content=f"TASK TO EXECUTE: {state['task'].description}")
        ] + recent_messages
        
        response = await llm_with_tools.ainvoke(final_messages)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"🛠️ Worker calling tools: {[tc['name'] for tc in response.tool_calls]}")
            
        # Returns the AI message back to the state
        return {"messages": [response]}

    return worker_agent_node, all_tools