import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage

# Import project modules
from main import build_graph
from Utils.Logger import get_logger

logger = get_logger("SERVER_API")

try:
    from Utils.database_manager import ltm_store
except ImportError:
    ltm_store = None

# Global state for MCP
mcp_session = None
mcp_tools = []
MCP_SERVER_URL = "http://localhost:8000/mcp/sse"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Establishes a persistent connection to the FastMCP server on boot."""
    global mcp_session, mcp_tools
    logger.info("🔌 Connecting to FastMCP Server...")
    try:
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                mcp_session = session
                logger.info(f"✅ FastMCP Connected! Loaded {len(mcp_tools)} tools.")
                yield
    except Exception as e:
        logger.warning(f"⚠️ FastMCP Connection skipped/failed: {e}")
        yield
    logger.info("🛑 Disconnected from FastMCP.")

app = FastAPI(title="Banking Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str | None = None
    action: str | None = None  # "approve" or "reject" for HITL

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Executes the graph and streams standard SSE tokens while logging the full response."""
    
    logger.info(f"💬 [INCOMING REQUEST] Thread: '{req.thread_id}' | User: '{req.user_id}' | Message: '{req.message}' | Action: '{req.action}'")

    async def generate():
        full_streamed_response = []
        
        async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
            graph = build_graph(all_mcp_tools=mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
            
            thread_config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}
            
            # Handle Human-in-the-Loop Resumes
            if req.action == "approve":
                inputs = None
            elif req.action == "reject":
                cancellation_msg = "Transaction cancelled by user."
                logger.info(f"🛑 [STREAM END] Thread: '{req.thread_id}' | Output: {cancellation_msg}")
                yield f"data: {cancellation_msg}\n\n"
                return
            else:
                inputs = {
                    "question": req.message,
                    "worker_responses": [],
                    "messages": [HumanMessage(content=req.message)]
                }

            # Stream execution
            try:
                stream_target = graph.astream_events(inputs, config=thread_config, version="v2")
                
                async for event in stream_target:
                    # Stream tokens from the aggregator node
                    if event["event"] == "on_chat_model_stream":
                        if event["metadata"].get("langgraph_node") == "aggregator":
                            chunk = event["data"]["chunk"].content
                            if chunk:
                                full_streamed_response.append(chunk)
                                chunk_clean = chunk.replace('\n', '\\n')
                                yield f"data: {chunk_clean}\n\n"
                
                # Check if graph paused for Human-In-The-Loop
                current_state = await graph.aget_state(thread_config)
                if current_state.next and "sensitive_tools_node" in current_state.next:
                    logger.info(f"⏸️ [STREAM INTERRUPTED] Thread: '{req.thread_id}' paused for HITL approval at 'sensitive_tools_node'.")
                    yield f"data: __INTERRUPT__\n\n"
                else:
                    # Log the full aggregated stream output
                    final_text = "".join(full_streamed_response)
                    logger.info(f"📢 [STREAM COMPLETE] Thread: '{req.thread_id}' | User: '{req.user_id}'\nFull Response Sent:\n{final_text}")

            except Exception as e:
                logger.error(f"❌ [STREAM ERROR] Thread: '{req.thread_id}' failed: {e}")
                yield f"data: System Error: {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/chat/history")
async def get_history(thread_id: str):
    """Fetches graph history so the frontend can render past messages."""
    async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
        graph = build_graph(all_mcp_tools=[], checkpointer=checkpointer, ltm_store=ltm_store)
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        
        if not state or not state.values:
            return {"messages": [], "is_paused": False}
            
        is_paused = bool(state.next and "sensitive_tools_node" in state.next)
        
        formatted_msgs = []
        for msg in state.values.get("messages", []):
            if getattr(msg, 'type', '') in ['human', 'ai'] and getattr(msg, 'content', ''):
                if msg.type == 'ai' and getattr(msg, 'tool_calls', None):
                    continue
                formatted_msgs.append({"role": "user" if msg.type == 'human' else "assistant", "content": msg.content})
                
        return {"messages": formatted_msgs, "is_paused": is_paused}