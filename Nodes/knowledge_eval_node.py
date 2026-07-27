from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from State.banking_state import BankingState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger

logger = get_logger("KNOWLEDGE_EVAL_NODE")

# 1. Strict Output Schema for the Evaluator
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# 2. The Evaluator Node
def evaluate_node(state: BankingState):
    logger.info("--- 🔍 EVALUATING KNOWLEDGE RETRIEVAL ---")
    question = state.get("question", "")
    messages = state.get("messages", [])
    
    # Get the output from the last executed tool
    if not messages:
        logger.warning("⚠️ No messages found in state during knowledge evaluation.")
        return {"relevance_score": "yes"}
        
    last_message = messages[-1]
    
    # If the last message isn't a tool output, just pass it through safely
    if last_message.type != "tool":
        logger.info("Last message is not a tool output. Defaulting relevance score to 'yes'.")
        return {"relevance_score": "yes"}
        
    retrieved_data = last_message.content
    logger.info(f"Evaluating retrieved data for question: '{question}'")
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict grader assessing the relevance of a retrieved document to a user question. "
                   "If the document contains keywords, policies, or semantic meaning that answers the question, grade it as 'yes'. "
                   "If the document is completely off-topic or empty, grade it as 'no'. Return ONLY 'yes' or 'no'."),
        ("human", f"User question: {question}\n\nRetrieved document: \n\n {retrieved_data}")
    ])
    
    evaluator_llm = fast_llm.with_structured_output(GradeDocuments)
    chain = grade_prompt | evaluator_llm
    
    try:
        score = chain.invoke({})
        logger.info(f"Relevance Score: {score.binary_score.upper()}")
        return {"relevance_score": score.binary_score}
    except Exception as e:
        logger.error(f"Evaluation failed, defaulting to 'yes': {e}")
        return {"relevance_score": "yes"}

# 3. The Rewriter Node
def rewrite_node(state: BankingState):
    logger.info("--- ✍️ REWRITING KNOWLEDGE QUERY ---")
    question = state.get("question", "")
    retries = state.get("knowledge_retries", 0)
    logger.info(f"Rewrite attempt #{retries + 1} for original question: '{question}'")
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at optimizing search queries for a banking vector database. "
                   "Look at the original question and output a refined, highly specific search query."),
        ("human", f"Original: {question}")
    ])
    
    try:
        rewriter_chain = rewrite_prompt | fast_llm
        new_query = rewriter_chain.invoke({}).content
        logger.info(f"New Query generated: '{new_query}'")
        
        # We append a HumanMessage instructing the agent to use the new query
        instruction = f"The previous search yielded irrelevant results. Search again using this optimized query: {new_query}"
        
        return {
            "messages": [HumanMessage(content=instruction)],
            "knowledge_retries": retries + 1
        }
    except Exception as e:
        logger.error(f"❌ Query rewriter failed: {e}")
        return {"knowledge_retries": retries + 1}