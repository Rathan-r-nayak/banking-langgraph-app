from State.banking_state import BankingState
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from Utils.Logger import get_logger

logger = get_logger("RECALL_NODE")

def recall_node(state: dict, config: RunnableConfig, store):
    """Fetches memories from the Store and injects them into the State."""
    logger.info("--- 🧠 RECALLING USER MEMORIES ---")
    
    # Get the user ID from the config (passed in from Streamlit)
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # Define the namespace where this user's memories live
    namespace = ("user", user_id, "facts")
    
    # Search the store for all saved items in this namespace
    try:
        saved_items = store.search(namespace)
        
        if not saved_items:
            logger.info(f"ℹ️ No previous memories found for user '{user_id}'.")
            return {"memories": "No previous memories found."}
        
        logger.info(f"✅ Fetched {len(saved_items)} memory fact(s) for user '{user_id}'.")
        # Format the extracted facts into a readable string
        memory_text = "\n".join([f"- {item.value['fact']}" for item in saved_items])
        
        return {"memories": memory_text}
    except Exception as e:
        logger.error(f"❌ Error searching memories for user '{user_id}': {e}")
        return {"memories": "No previous memories found."}