# Nodes/remember_node.py
import uuid
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from State.banking_state import BankingState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger

logger = get_logger("REMEMBER_NODE")

# 🌟 1. Define the exact JSON schema we want
class MemoryFact(BaseModel):
    fact: str = Field(description="A distinct, standalone fact about the user's finances or preferences.")

class MemoryExtraction(BaseModel):
    facts: List[MemoryFact] = Field(
        description="List of extracted facts. Empty if nothing new is learned.", 
        default_factory=list
    )

def remember_node(state: BankingState, config: RunnableConfig, store: BaseStore):
    """Intelligently extracts long-term banking facts using strict Pydantic schemas."""
    logger.info("--- 💾 RUNNING REMEMBER NODE (LTM EXTRACTION) ---")
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    # Grab the last 4 messages to catch the user's request and the AI's final answer
    recent_messages = messages[-4:]
    transcript = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in recent_messages])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise data extraction assistant."),
        ("human", (
            "Analyze the conversation below. Extract any durable, long-term personal facts about the user "
            "(e.g., salary, financial goals, preferred accounts, risk appetite, or recurring financial constraints) "
            "that would be useful to remember for future banking sessions.\n\n"
            "Transcript:\n{transcript}"
        ))
    ])
    
    # 🌟 2. Force the fast LLM (e.g., GPT-4o-mini) to output the Pydantic schema
    chain = prompt | fast_llm.with_structured_output(MemoryExtraction)
    
    try:
        # Returns a clean Python object, no manual JSON parsing needed!
        extraction: MemoryExtraction = chain.invoke({"transcript": transcript})
        
        saved_count = 0
        for item in extraction.facts:
            store.put(
                namespace, 
                str(uuid.uuid4()), 
                {"fact": item.fact}
            )
            logger.info(f"💾 [LTM Saved] -> {item.fact}")
            saved_count += 1
            
        if saved_count == 0:
            logger.info("ℹ️ No new long-term facts extracted.")
            
    except Exception as e:
        logger.error(f"❌ Failed to extract memory: {e}")
        
    return {}