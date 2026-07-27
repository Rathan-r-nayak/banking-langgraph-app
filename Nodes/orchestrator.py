from langchain_core.prompts import ChatPromptTemplate

from State import banking_state
from Schema.task import Task, TaskPlan
from Utils.Helpers import format_chat_history
from Utils.Logger import get_logger
from Config.llm_config import primary_llm
from Config.llm_config import fast_llm

logger = get_logger("ORCHESTRATOR")

# 1. Update the System Prompt to act as a routing directory
ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator for a secure Banking Assistant.
Your job is to analyze the user's request, break it down into independent tasks, and assign EACH task to the correct specialized agent.

AVAILABLE AGENTS:
- 'account_agent': Assign here for checking account balances, fetching account details, opening new accounts, and processing direct deposits or withdrawals.
- 'transaction_agent': Assign here for transferring money between accounts or sending money to other people.
- 'knowledge_agent': Assign here for general inquiries, banking policies, interest rates, loans, or FAQs.

Rules:
1. Break down multi-part requests into separate tasks (e.g., "What is my balance and what are mortgage rates?" becomes one task for account_agent and one for knowledge_agent).
2. Make the task descriptions highly specific so the downstream agent knows exactly what tool to use.
3. You MUST assign one of the exact agent names listed above to handle the task.
"""

def orchestrator_node(state: banking_state):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR: PLANNING TASKS ---")
    question = state.get("question", "")
    
    raw_messages = state.get("messages", [])
    chat_history_text = format_chat_history(raw_messages[:-1])
    
    structured_llm = fast_llm.with_structured_output(TaskPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = prompt | structured_llm

    try:
        plan: TaskPlan = chain.invoke({"question": question})
        logger.info(f"Orchestrator created {len(plan.tasks)} tasks.")
        
        # Save the planned tasks to the state
        agent_type = getattr(plan.tasks[0], "agent", None) or getattr(plan.tasks[0], "type", "unknown")
        logger.info(f"Assigned agent: {agent_type}")
        return {"tasks": plan.tasks}
        
    except Exception as e:
        logger.error(f"Orchestrator failed to generate plan: {e}")
        # Failsafe: Ensure the fallback task explicitly names an agent
        fallback_task = Task(
            task_id="fallback_1", 
            description=question,
            objective="Handle general fallback query",
            agent="knowledge_agent" # or "type='knowledge_agent'" depending on your Pydantic schema
        )
        return {"tasks": [fallback_task]}