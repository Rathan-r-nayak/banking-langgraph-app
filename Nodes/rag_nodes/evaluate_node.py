# Nodes/rag_nodes.py (continued)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from Config.llm_config import fast_llm

from State.rag_state import RagState
from Utils.Logger import get_logger
# Import your local DB connections (e.g., Neo4j, Milvus/FAISS/Chroma)
# from DB.vector_store import get_vector_store
# from DB.graph_store import get_graph_store

logger = get_logger("EVALUATE_NODE")


def evaluate_node(state: RagState):
    logger.info("--- 🔍 EVALUATING KNOWLEDGE RETRIEVAL ---")
    question = state.get("question", "")
    documents = state.get("documents", "")

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict grader assessing document relevance. Respond in strictly valid JSON: {{\"binary_score\": \"yes\"}} or {{\"binary_score\": \"no\"}}."),
        ("human", "User question: {question}\n\nRetrieved document:\n{documents}")
    ])

    chain = grade_prompt | fast_llm | JsonOutputParser()

    try:
        score_dict = chain.invoke({"question": question, "documents": documents})
        score = score_dict.get("binary_score", "yes").lower()
        logger.info(f"Relevance Score: {score.upper()}")
        return {"relevance_score": score}
    except Exception as e:
        logger.error(f"Evaluation failed, defaulting to 'yes': {e}")
        return {"relevance_score": "yes"}


