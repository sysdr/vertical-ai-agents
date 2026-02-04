from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from datetime import datetime
import json
import logging
import os

from dotenv import load_dotenv
from pathlib import Path

from tools import ToolRegistry, WikipediaTool, StockPriceTool, CalculatorTool, WeatherTool
from agent import ReActAgent

# Load .env from backend/ or project root (Docker uses env_file from compose dir)
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ReAct Agent API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini - use GEMINI_API_KEY from env (or .env file)
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY or API_KEY.strip() in ("", "your_api_key_here"):
    raise ValueError(
        "GEMINI_API_KEY not set or invalid. Create a .env file with GEMINI_API_KEY=your_key "
        "or export GEMINI_API_KEY. Get a key at https://aistudio.google.com/apikey. "
        "See API_KEY_SETUP.md for instructions."
    )
# Use REST transport - more reliable than gRPC for long requests (avoids 504 Deadline Exceeded)
genai.configure(api_key=API_KEY, transport="rest")

# Initialize tool registry
tool_registry = ToolRegistry()
tool_registry.register(WikipediaTool())
tool_registry.register(StockPriceTool())
tool_registry.register(CalculatorTool())
tool_registry.register(WeatherTool())

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 5
    session_id: Optional[str] = None

class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: str

class QueryResponse(BaseModel):
    success: bool
    result: str
    reasoning_trace: List[ReasoningStep]
    iterations_used: int
    session_id: str
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

# In-memory session storage (use Redis in production)
sessions = {}
total_queries_count = 0

@app.get("/")
def read_root():
    return {
        "service": "ReAct Agent API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/agent/query", "/agent/tools", "/agent/history/{session_id}"]
    }

@app.post("/agent/query", response_model=QueryResponse)
async def execute_agent_query(request: QueryRequest):
    """Execute a ReAct agent query with full reasoning trace"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create agent instance
        agent = ReActAgent(
            tool_registry=tool_registry,
            api_key=API_KEY,
            max_iterations=request.max_iterations
        )
        
        logger.info(f"Processing query: {request.query} [Session: {session_id}]")
        
        global total_queries_count
        total_queries_count += 1
        
        # Execute agent
        result = agent.execute(request.query)
        
        # Store session
        sessions[session_id] = {
            "query": request.query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Format reasoning trace
        reasoning_trace = []
        for i, step in enumerate(result['reasoning_trace'], 1):
            reasoning_trace.append(ReasoningStep(
                step_number=i,
                thought=step.get('thought', ''),
                action=step.get('action', ''),
                action_input=step.get('action_input', {}),
                observation=step.get('observation', ''),
                timestamp=step.get('timestamp', datetime.now().isoformat())
            ))
        
        return QueryResponse(
            success=result['success'],
            result=result['final_answer'],
            reasoning_trace=reasoning_trace,
            iterations_used=result['iterations'],
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/tools", response_model=List[ToolInfo])
def list_tools():
    """List all available tools with their schemas"""
    tools = []
    for tool_name, tool in tool_registry.list_tools().items():
        schema = tool.schema
        tools.append(ToolInfo(
            name=tool_name,
            description=schema['description'],
            parameters=schema['parameters']
        ))
    return tools

@app.get("/agent/history/{session_id}")
def get_session_history(session_id: str):
    """Retrieve reasoning trace for a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tools_loaded": len(tool_registry.list_tools()),
        "active_sessions": len(sessions),
        "total_queries": total_queries_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
