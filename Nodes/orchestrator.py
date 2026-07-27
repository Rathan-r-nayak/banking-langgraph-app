from langchain_core.prompts import ChatPromptTemplate

from State import banking_state
from Schema.task import Task, TaskPlan
from Utils.Logger import get_logger
from Config.llm_config import primary_llm


logger = get_logger("ORCHESTRATOR")

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator for a secure Banking Assistant.
Your job is to analyze the user's request and break it down into independent, parallelizable tasks.

Rules:
1. If the user asks for multiple distinct things (e.g., "What is my account balance and what are your current mortgage rates?"), create separate tasks for each.
2. Make the task descriptions highly specific so a downstream worker agent knows exactly what tool to use or what information to retrieve.
3. If the query requires a single action, output a list with exactly one task.
"""

def orchestrator_node(state: banking_state):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR: PLANNING TASKS ---")
    question = state.get("question", "")
    
    structured_llm = primary_llm.with_structured_output(TaskPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = prompt | structured_llm

    try:
        plan: TaskPlan = chain.invoke({"question": question})
        logger.info(f"Orchestrator created {len(plan.tasks)} tasks.")
        
        # Save the planned tasks to the state
        return {"tasks": plan.tasks}
        
    except Exception as e:
        logger.error(f"Orchestrator failed to generate plan: {e}")
        # Failsafe: Create a single task containing the raw question
        fallback_task = Task(task_id="fallback_1", description=question)
        return {"tasks": [fallback_task]}