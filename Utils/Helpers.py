import base64
import mimetypes
from langchain_core.messages import HumanMessage
from Utils.Logger import get_logger
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore


# Import your primary model for vision tasks
from Config.llm_config import primary_llm 

logger = get_logger("HELPERS")

def format_chat_history(messages) -> str:
    """
    Converts a list of LangChain messages into a clean, readable string.
    This acts as the 'Short-Term Memory' for the Orchestrator and Workers.
    """
    if not messages:
        return "No previous conversation."
    
    recent_msgs = messages[-6:]
    formatted = []
    
    for msg in recent_msgs:
        # Handle LangChain Message Objects
        if hasattr(msg, "type"):
            role = "User" if msg.type == "human" else "AI"
            content = msg.content
        # Handle Raw Dictionaries (Streamlit fallback)
        elif isinstance(msg, dict):
            role = "User" if msg.get("role") == "user" else "AI"
            content = msg.get("content", "")
        else:
            continue
            
        formatted.append(f"{role}: {content}")
        
    return "\n".join(formatted)


def fetch_user_ltm(config: RunnableConfig, store: BaseStore) -> str:
    """
    Safely fetches and formats Long-Term Memory (LTM) facts for a user from the BaseStore.
    """
    if store is None:
        return "No known facts."
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "details")

    try:
        user_memory = store.search(namespace)
        
        if user_memory:
            # Safely extract the data depending on how it was saved (handles strings or lists)
            facts_list = []
            for item in user_memory:
                # Adjust 'data' to 'facts' if you saved your dictionary key as {"facts": [...]}
                val = item.value.get("data", item.value.get("facts", "")) 
                if isinstance(val, list):
                    facts_list.extend(val)
                else:
                    facts_list.append(str(val))
                    
            long_term_facts = "\n".join(facts_list)
            logger.info(f"Loaded LTM for '{user_id}'")
            return long_term_facts
            
    except Exception as e:
        logger.error(f"Error fetching LTM: {e}")
        
    logger.info(f"No existing LTM found for '{user_id}'")
    return "No previous memory."




def analyze_image_context(image_path: str) -> str:
    """
    Uses the Vision LLM to analyze IT-specific images (errors, logs, diagrams).
    Extracts text, error codes, and system states to inject into the text prompt.
    """
    try:
        # 1. Dynamically detect mime type so PNGs and JPGs both work cleanly
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"

        # 2. Encode the image to Base64 format
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 3. Build the multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        "You are an L3 Technical Support Vision AI. Analyze this image. "
                        "1. If it's an error screen/terminal, extract the exact error codes and text. "
                        "2. If it's a software UI, describe the application state and any visible issues. "
                        "3. If it's a network/architecture diagram, list the connected components. "
                        "Focus entirely on actionable technical context. Do not describe aesthetic elements."
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                }
            ]
        )
        
        logger.info("Analyzing attached image for technical context...")
        
        # 4. Invoke the model
        response = primary_llm.invoke([message])
        return response.content
        
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        return "System failed to analyze the attached image."