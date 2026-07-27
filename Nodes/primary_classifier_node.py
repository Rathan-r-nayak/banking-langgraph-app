from langchain_classic.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from State.banking_state import BankingState
from Config.llm_config import primary_llm
from Schema.primary_classifier import PrimaryClassifierDecision

TRIAGE_SYSTEM_PROMPT = """You are the first line of defense for a secure Banking Assistant.
Your job is to evaluate the user's latest message and route it appropriately.

Rules:
1. If the user asks about banking policies, transactions, account status, or requires complex reasoning, set 'is_workflow_required' to True and leave 'direct_response' empty.
2. If the user sends a standard greeting (e.g., "Hi", "How are you?"), set 'is_workflow_required' to False and provide a polite, brief greeting in 'direct_response' offering banking assistance.
3. If the user asks an out-of-domain question (e.g., coding, recipes, general trivia), set 'is_workflow_required' to False and politely explain that you can only assist with banking-related queries.
"""
from Utils.Logger import get_logger

logger = get_logger("Primary Classifier")

def triage_router(state: BankingState):
    structured_llm = primary_llm.with_structured_output(PrimaryClassifierDecision)

    question = state.get("question", "")
    logger.info(f"🗣️ USER REQ : {question}")
    
    logger.info("--- 🛡️ RUNNING INTENT ROUTER & GATEKEEPER CHECK ---")

    if not question:
        logger.warning("Empty question received in state.")
        return {"requires_workflow": False, "generation": "How can I help you today?"}
    
    logger.info(f"User Query: '{question}'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = prompt | structured_llm

    try:
        decision: PrimaryClassifierDecision = chain.invoke({"question": question})
        logger.info(f"Decision Result: Action Required = {decision.is_workflow_required}")

        if decision.is_workflow_required:
            return {
                "requires_workflow": True
            }
        else:
            new_message = AIMessage(content=decision.direct_response)
            return {
                "requires_workflow": False,
                "messages": [new_message],
                "generation": decision.direct_response
            }
    except Exception as e:
        logger.error(f"Primary classifier LLM extraction failed: {e}")
        logger.warning("Failsafe triggered: Defaulting to Action pipeline.")
        return {
            "requires_workflow": True
        }