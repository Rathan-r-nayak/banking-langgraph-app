# Nodes/orchestrator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from State.banking_state import BankingState
from Schema.task import OrchestratorPlan
from Utils.helpers import format_chat_history, get_short_term_memory
from Utils.Logger import get_logger
from Config.llm_config import primary_llm

logger = get_logger("ORCHESTRATOR")

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Lead Orchestrator for a secure Banking Assistant.
Your job is to break the user's request into a list of parallel tasks.

User Profile / Long-Term Memory:
{user_memories}

Rules:
1. If you need data, set is_workflow_complete to False and generate a list of Tasks. 
2. Break independent requests into separate tasks (e.g., Task 1: Check balance, Task 2: Check loan rates).
3. If the chat history already contains the answers you need, set is_workflow_complete to True and provide the final_answer.
"""

def orchestrator_node(state: BankingState):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR: PLANNING TASKS ---")
    memories = state.get("memories", "No known facts.")
        
    question = state.get("question", "")
    recent_messages = get_short_term_memory(state.get("messages", []), k=6)
    chat_history_text = format_chat_history(recent_messages)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "Conversation History:\n{chat_history}\n\nRequest: {question}")
    ])

    # 🌟 Force Pydantic output!
    chain = prompt | primary_llm.with_structured_output(OrchestratorPlan)
        

    try:
        plan: OrchestratorPlan = chain.invoke({
            "user_memories": memories,
            "chat_history": chat_history_text,
            "question": question,
        })
        
        if plan.is_workflow_complete:
            logger.info("✅ Workflow complete.")
            return {
                "is_workflow_complete": True, 
                "generation": plan.final_answer,
                "messages": [AIMessage(content=plan.final_answer)]
            }
        else:
            logger.info(f"👷 Created {len(plan.tasks)} parallel tasks.")
            return {
                "is_workflow_complete": False, 
                "tasks": plan.tasks
            }

    except Exception as e:
        logger.error(f"Orchestrator plan failed: {e}")
        return {"is_workflow_complete": True, "generation": "I encountered an error."}