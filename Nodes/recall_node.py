from State.banking_state import BankingState
from langgraph.store.memory import InMemoryStore


def recall_node(state: BankingState, config: dict, store: InMemoryStore):
    """Fetches memories from the Store and injects them into the State."""
    
    # Get the user ID from the config (passed in from Streamlit)
    user_id = config["configurable"].get("user_id", "default_user")
    
    # Define the namespace where this user's memories live
    namespace = ("user", user_id, "facts")
    
    # Search the store for all saved items in this namespace
    saved_items = store.search(namespace)
    
    if not saved_items:
        return {"memories": "No previous memories found."}
    
    # Format the extracted facts into a readable string
    memory_text = "\n".join([f"- {item.value['fact']}" for item in saved_items])
    
    return {"memories": memory_text}