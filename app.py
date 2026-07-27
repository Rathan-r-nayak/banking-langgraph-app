import sqlite3
import streamlit as st
import uuid
import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Imports from your project
from Utils.HistoryLoaders import load_chat_history
from main import build_graph

# -----------------------------------------------------------------------------
# Import your SQLite LTM Store 
# Note: Ensure you have your SqliteKeyValueStore initialized in this imported file!
# -----------------------------------------------------------------------------
try:
    from Utils.database_manager import ltm_store
except ImportError:
    # Fallback: If you defined ltm_store inside main.py instead, change this to:
    # from main import ltm_store
    try:
        from main import ltm_store  # If ltm_store is initialized inside main.py
    except ImportError:
        ltm_store = None
    ltm_store = None 
    st.warning("⚠️ LTM Store could not be imported. Please verify your database manager file.")


MCP_SERVER_URL = "http://localhost:8000/mcp/sse"

# -----------------------------------------------------------------------------
# 1. Page Configuration & Session Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Banking Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Secure Banking Agent")

# Initialize persistent SQLite Checkpointer
if "checkpointer" not in st.session_state:
    # Connect to the same DB your terminal uses
    sqlite_conn = sqlite3.connect("banking_checkpoints.db", check_same_thread=False)
    st.session_state.checkpointer = SqliteSaver(sqlite_conn)

# Hardcode a thread ID for testing, or tie this to a user login system later
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_thread_123" 
    st.session_state.user_id = "rathan_123"

thread_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "user_id": st.session_state.user_id
    }
}

# Initialize UI chat messages history & Load Past History (STM)
if "messages" not in st.session_state:
    with st.spinner("Loading previous conversation..."):
        # Temporary app instance just to read the history (no tools needed yet)
        temp_app = build_graph(
            all_mcp_tools=[], 
            checkpointer=st.session_state.checkpointer,
            ltm_store=ltm_store
        )
        
        # Call your history loader
        past_messages = load_chat_history(temp_app, st.session_state.thread_id)
        st.session_state.messages = past_messages if past_messages else []


# -----------------------------------------------------------------------------
# 2. Async Graph Execution Helper
# -----------------------------------------------------------------------------
async def run_agent_turn(inputs, thread_config):
    """Establishes MCP connection, builds graph, and executes a turn or resumes."""
    async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Fetch tools dynamically from FastMCP server
            mcp_tools = await load_mcp_tools(session)
            
            # Compile graph with persistent checkpointer AND LTM Store
            app = build_graph(
                all_mcp_tools=mcp_tools, 
                checkpointer=st.session_state.checkpointer,
                ltm_store=ltm_store
            )
            
            # Run or resume the graph
            result = app.invoke(inputs, config=thread_config)
            return app, result

def execute_turn(inputs):
    return asyncio.run(run_agent_turn(inputs, thread_config))

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Session Settings")
    st.caption(f"Thread ID:\n`{st.session_state.thread_id}`")
    
    if st.button("🔄 Clear Chat / New Session", use_container_width=True):
        # Generate a new thread ID to start fresh
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 4. Render Conversation History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    # Handle LangChain message objects
    if hasattr(msg, 'type'):
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content
    # Handle dictionary fallback from history loader
    elif isinstance(msg, dict):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
    else:
        continue
        
    with st.chat_message(role):
        st.markdown(content)

# -----------------------------------------------------------------------------
# 5. Check State for Interrupts (Human-In-The-Loop Approval)
# -----------------------------------------------------------------------------
app_instance = None
try:
    dummy_checkpointer = st.session_state.checkpointer
    temp_state = dummy_checkpointer.get(thread_config)
    next_node = temp_state.get("next", ()) if temp_state else ()
except Exception as e:
    import traceback
    st.error(f"Execution Error: {e}")
    # This will print the raw, unhidden error trace to the UI
    with st.expander("Show detailed error trace"):
        st.code(traceback.format_exc())
    next_node = ()

# Check if graph is paused waiting for approval at sensitive_tools_node
if next_node and "sensitive_tools_node" in next_node:
    st.warning("⚠️ **Action Required:** This transaction requires human authorization.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Transfer", type="primary", use_container_width=True):
            with st.spinner("Processing approved transaction..."):
                app_instance, result = execute_turn(inputs=None)
                
                final_answer = result.get("generation", "Transaction completed successfully.")
                st.session_state.messages.append(AIMessage(content=final_answer))
                st.rerun()
                
    with col2:
        if st.button("❌ Reject / Cancel", use_container_width=True):
            st.session_state.messages.append(AIMessage(content="Transaction was cancelled by user."))
            # Assigning a new thread ID effectively drops the paused state
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

# -----------------------------------------------------------------------------
# 6. Standard User Chat Input
# -----------------------------------------------------------------------------
else:
    if prompt := st.chat_input("Ask about your accounts, policies, or initiate transfers..."):
        
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        input_state = {
            "question": prompt,
            "worker_responses": []
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    app_instance, result = execute_turn(input_state)
                    
                    current_state = app_instance.get_state(thread_config)
                    
                    if current_state.next and "sensitive_tools_node" in current_state.next:
                        st.session_state.messages.append(
                            AIMessage(content="⚠️ Transfer pending approval. Please review the details above.")
                        )
                    else:
                        final_answer = result.get("generation", "Response generated.")
                        st.session_state.messages.append(AIMessage(content=final_answer))
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    import traceback
                    st.error(f"Execution Error: {e}")
                    # This will print the raw, unhidden error trace to the UI
                    with st.expander("Show detailed error trace"):
                        st.code(traceback.format_exc())