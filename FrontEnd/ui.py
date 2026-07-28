import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

# Import your graph builder and tools
from main import build_graph

st.set_page_config(page_title="Secure Banking AI", page_icon="🏦", layout="centered")

# --- 1. Initialize LangGraph & Checkpointer ---
# We use st.cache_resource so the graph compiles only once
@st.cache_resource
def init_graph():
    # In production, replace MemorySaver with your PostgreSQL/SQLite checkpointer
    checkpointer = MemorySaver()
    # Pass your actual MCP tools and LTM store here
    return build_graph(all_mcp_tools=[], checkpointer=checkpointer, ltm_store=None)

app = init_graph()

# --- 2. Session Setup ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "dev_session_001"

# The config dictates which database thread LangGraph pulls from
config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "rathan"}}

# --- 3. Render Chat History ---
def render_history():
    """Reads directly from LangGraph's state database to render the UI."""
    state = app.get_state(config)
    if not state.values:
        return
        
    for msg in state.values.get("messages", []):
        # Render Human Messages
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
            
        # Render AI Messages (Hide internal reasoning/tool calls from the user)
        elif isinstance(msg, AIMessage):
            if msg.content and not msg.tool_calls:
                st.chat_message("assistant").write(msg.content)

st.title("🏦 Secure Banking Orchestrator")
render_history()

# --- 4. Evaluate Graph State (The Gatekeeper) ---
current_state = app.get_state(config)
is_paused = len(current_state.next) > 0

# If the graph is paused, we lock the chat input and display action buttons
if is_paused:
    last_msg = current_state.values["messages"][-1]
    
    # --- HITL SCENARIO A: Web Search Approval (RAG Subgraph) ---
    if isinstance(last_msg, AIMessage) and "Would you like me to search the web?" in last_msg.content:
        st.warning("🌐 **Web Search Authorization Required**")
        st.write("Internal policies yielded no results. Permit external search?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, search the web", use_container_width=True):
                # Inject the user's "Yes" into the state and resume the graph
                app.update_state(config, {"messages": [HumanMessage(content="Yes")]})
                app.invoke(None, config) # Trigger resume
                st.rerun()
        with col2:
            if st.button("❌ No, use internal only", use_container_width=True):
                # Inject "No" and resume
                app.update_state(config, {"messages": [HumanMessage(content="No")]})
                app.invoke(None, config)
                st.rerun()
                
    # --- HITL SCENARIO B: Sensitive Tools (Worker Subgraph) ---
    elif isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        st.error("🔒 **Sensitive Transaction Pending Approval**")
        
        # Display the pending parallel tool calls
        for tool in last_msg.tool_calls:
            st.info(f"**Action:** `{tool['name']}`\n\n**Payload:** `{tool['args']}`")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve Transaction", use_container_width=True):
                # Resume execution; LangGraph will immediately execute the pending tools
                app.invoke(None, config)
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                # Create rejection messages for every pending tool call
                rejections = [
                    ToolMessage(
                        tool_call_id=tc["id"], 
                        name=tc["name"], 
                        content="SYSTEM: User rejected this transaction."
                    )
                    for tc in last_msg.tool_calls
                ]
                # Inject the failure directly into the tool node to safely collapse the execution
                app.update_state(config, {"messages": rejections}, as_node="sensitive_tools_node")
                app.invoke(None, config)
                st.rerun()

# --- 5. Standard Chat Input ---
# Only display the input bar if the graph is currently idle
else:
    if prompt := st.chat_input("Ask about policies or request a transaction..."):
        st.chat_message("user").write(prompt)
        
        with st.spinner("Orchestrating workflow..."):
            # We pass both 'question' for the State dict and 'messages' for the history
            app.invoke({
                "question": prompt, 
                "messages": [HumanMessage(content=prompt)]
            }, config)
            
            st.rerun()