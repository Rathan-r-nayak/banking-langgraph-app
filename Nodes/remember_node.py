import json
import uuid
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from State.banking_state import BankingState

def remember_node(state: BankingState, config: dict, store: InMemoryStore):
    """Uses Ollama to intelligently extract long-term banking facts from the conversation."""
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    # Grab the recent conversation history (last few messages for context)
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    # Format the last couple of messages into a transcript block for the LLM
    transcript = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[-4:]])
    
    # Initialize your local Ollama model
    # (Using your local Qwen Coder model since it excels at JSON parsing)
    llm = ChatOllama(
        model="qwen-2.5.1-coder-it:latest", 
        temperature=0
    )
    
    extraction_prompt = f"""Analyze the conversation below. Extract any durable, long-term personal facts about the user (e.g., salary, financial goals, preferred accounts, risk appetite, or recurring financial constraints) that would be useful to remember for future banking sessions.

    If there are no new facts worth remembering, return an empty JSON array: []

    If there are facts, return a valid JSON array of objects with a single key "fact". Example format:
    [
    { "fact": "User's monthly salary is 50000" },
    { "fact": "User prefers low-risk investment options" }
    ]

    Conversation Transcript:
    {transcript}

    Return ONLY the valid JSON array, no extra markdown formatting or conversational text.
    """

    try:
        # Call the local LLM
        response = llm.invoke([
            SystemMessage(content="You are a precise data extraction assistant. You output strictly valid JSON."),
            HumanMessage(content=extraction_prompt)
        ])
        
        content = response.content.strip()
        
        # Clean up any accidental markdown code blocks (e.g., ```json ... ```)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse the JSON response
        facts_list = json.loads(content)
        
        # Save each extracted fact into the LangGraph Store
        for item in facts_list:
            fact_text = item.get("fact")
            if fact_text:
                store.put(
                    namespace, 
                    str(uuid.uuid4()), 
                    {"fact": fact_text}
                )
                print(f"💾 [LTM Saved] -> {fact_text}")
                
    except Exception as e:
        print(f"❌ Failed to extract memory via Ollama: {e}")
        
    return {}