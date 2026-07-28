from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from State.banking_state import BankingState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger
from helpers import format_chat_history, get_short_term_memory

logger = get_logger("Primary Classifier")

# 1. Define the strictly typed output schema
class TriageDecision(BaseModel):
    is_workflow_required: bool = Field(
        description="True if the request needs banking tools or complex logic. False for simple greetings or chit-chat."
    )
    direct_response: str = Field(
        description="If workflow is not required, provide the direct conversational response here.", 
        default="How can I help you?"
    )

TRIAGE_SYSTEM_PROMPT = """
You are the first line of defense for a secure Banking Assistant.
Your job is to decide if the user's request requires the full banking workflow (tools, orchestration) or if it can be answered directly (e.g., greetings, pleasantries).

User Profile / Long-Term Memory:
{user_memories}

Rules:
1. If the user greets you, use their name or profile details from the Long-Term Memory to personalize the 'direct_response'.
2. If the user asks about banking data, policies, or transactions, set 'is_workflow_required' to True.
3. Do not attempt to answer banking questions directly. Always route them to the workflow.
"""

def triage_router(state: BankingState):
    question = state.get("question", "")
    
    # 2. Extract LTM from the state (populated by recall_node)
    memories = state.get("memories", "No known facts.")
    
    logger.info(f"🗣️ USER REQ : {question}")
    logger.info("--- 🛡️ RUNNING INTENT ROUTER & GATEKEEPER CHECK ---")

    if not question:
        return {
            "requires_workflow": False,
            "generation": "How can I help you today?",
            "worker_responses": [],
        }

    # 3. Extract STM so the router understands context
    recent_messages = get_short_term_memory(state.get("messages", []), k=4)
    chat_history = format_chat_history(recent_messages)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", "Chat History:\n{chat_history}\n\nUser Request: {question}"),
    ])

    # 4. Use structured output for guaranteed schema compliance
    chain = prompt | fast_llm.with_structured_output(TriageDecision)

    try:
        decision: TriageDecision = chain.invoke({
            "question": question,
            "user_memories": memories,
            "chat_history": chat_history
        })

        logger.info(f"Decision Result: {decision}")

        if decision.is_workflow_required:
            return {
                "requires_workflow": True,
                "worker_responses": [],
            }

        return {
            "requires_workflow": False,
            "messages": [AIMessage(content=decision.direct_response)],
            "generation": decision.direct_response,
            "worker_responses": [],
        }

    except Exception as e:
        logger.error(f"Primary classifier LLM extraction failed: {e}")
        logger.warning("Failsafe triggered: Defaulting to Action pipeline.")

        return {
            "requires_workflow": True,
            "worker_responses": [],
        }