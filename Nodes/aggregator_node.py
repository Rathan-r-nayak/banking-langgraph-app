from State.banking_state import BankingState
from Utils.Logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from Config.llm_config import primary_llm
from langchain.messages import AIMessage


logger = get_logger("AGGREGATOR")

AGGREGATOR_SYSTEM_PROMPT = """You are the final response generator for a secure Banking Assistant.
Your system has completed several internal tasks to gather information for the user.

Your job is to synthesize these internal worker results into a single, cohesive, and user-friendly response that directly answers the user's original query.

Rules:
1. Do NOT mention "workers", "tasks", or internal processes to the user. Present the information as if you knew it instantly.
2. Aggregate the data logically. Use Markdown tables, lists, or bold text for readability.
3. If any worker reported an error or failed to find information, politely apologize and explain what specific information could not be retrieved.
"""

def aggregator_node(state: BankingState):
    question = state.get("question", "")
    # The state reducer (operator.add) has combined all parallel outputs into this list
    worker_responses_list = state.get("worker_responses", [])
    
    combined_worker_data = "\n\n".join(worker_responses_list)
    
    if not combined_worker_data:
        logger.warning("No worker responses found. Returning default error.")
        error_msg = "I'm sorry, but I was unable to process your request at this time."
        return {
            "generation": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }
    logger.info(f"The response of the worker is {combined_worker_data}")
    prompt = ChatPromptTemplate.from_messages(AGGREGATOR_SYSTEM_PROMPT)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATOR_SYSTEM_PROMPT),
        ("human", "Original User Query: {question}\n\nInternal Data Gathered:\n{worker_data}")
    ])
    chain = prompt | primary_llm
    
    try:
        response = chain.invoke({
            "question": question,
            "worker_data": combined_worker_data
        })
        
        final_answer = response.content
        logger.info("✅ Aggregator successfully generated the final response.")
        
        return {
            "generation": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }
    except Exception as e:
        logger.error(f"Aggregator failed: {e}")
        fallback = "An error occurred while formatting your final response."
        return {
            "generation": fallback,
            "messages": [AIMessage(content=fallback)]
        }