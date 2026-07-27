import streamlit as st
import uuid
import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

# Import your graph builder from main.py
from main import build_graph

MCP_SERVER_URL = "http://localhost:8000/sse"

# -----------------------------------------------------------------------------
# 1. Page Configuration & Session Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Banking Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Secure Banking Agent")

# Initialize unique thread ID for checkpointer tracking
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize UI chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent in-memory checkpointer across Streamlit reruns
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

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
            
            # Compile graph with persistent checkpointer
            app = build_graph(mcp_tools, checkpointer=st.session_state.checkpointer)
            
            # Run or resume the graph
            result = await app.ainvoke(inputs, config=thread_config)
            return app, result

# Helper to run async functions inside Streamlit safely
def execute_turn(inputs):
    return asyncio.run(run_agent_turn(inputs, thread_config))

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Session Settings")
    st.caption(f"Thread ID:\n`{st.session_state.thread_id}`")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 4. Render Conversation History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# -----------------------------------------------------------------------------
# 5. Check State for Interrupts (Human-In-The-Loop Approval)
# -----------------------------------------------------------------------------
# Inspect graph state via a quick connection check or saved checkpoint state
app_instance = None
try:
    # Build temporary graph instance to inspect current thread state
    dummy_checkpointer = st.session_state.checkpointer
    # Quick check on graph state
    temp_state = dummy_checkpointer.get(thread_config)
    next_node = temp_state.get("next", ()) if temp_state else ()
except Exception:
    next_node = ()

# If graph is paused waiting for approval at sensitive_tools_node
if next_node and "sensitive_tools_node" in next_node:
    st.warning("⚠️ **Action Required:** This transaction requires human authorization.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Transfer", type="primary", use_container_width=True):
            with st.spinner("Processing approved transaction..."):
                # Resume execution by passing inputs=None
                app_instance, result = execute_turn(inputs=None)
                
                final_answer = result.get("generation", "Transaction completed successfully.")
                st.session_state.messages.append(AIMessage(content=final_answer))
                st.rerun()
                
    with col2:
        if st.button("❌ Reject / Cancel", use_container_width=True):
            st.session_state.messages.append(AIMessage(content="Transaction was cancelled by user."))
            # Reset session thread or handle cancellation state
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

# -----------------------------------------------------------------------------
# 6. Standard User Chat Input
# -----------------------------------------------------------------------------
else:
    if prompt := st.chat_input("Ask about your accounts, policies, or initiate transfers..."):
        
        # Display user query in chat
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Standard state input format
        input_state = {
            "question": prompt,
            "worker_responses": []
        }

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    app_instance, result = execute_turn(input_state)
                    
                    # Check if execution got interrupted during invocation
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