from dotenv import load_dotenv
load_dotenv()  # Load before agent import so GEMINI_API_KEY is available

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from agent.tool_equipped_agent import ToolEquippedAgent
from tools.registry import ToolRegistry
from tools.weather import WeatherTool
from tools.calculator import CalculatorTool
from tools.database import DatabaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L28 Tool-Equipped Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tool registry
tool_registry = ToolRegistry()
tool_registry.register(WeatherTool())
tool_registry.register(CalculatorTool())
tool_registry.register(DatabaseTool())

# Initialize agent
agent = ToolEquippedAgent(tool_registry=tool_registry)

class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    include_metrics: bool = True

class QueryResponse(BaseModel):
    response: str
    intent_type: str
    tool_used: Optional[str] = None
    sources: List[str] = []
    metrics: Optional[Dict[str, Any]] = None
    conversation_id: str

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Process query through tool-equipped agent"""
    try:
        result = await agent.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            include_metrics=request.include_metrics
        )
        return result
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        err_msg = str(e).lower()
        # Return 401 for API key errors
        api_key_indicators = ["api_key_invalid", "api key expired", "api key invalid", "renew the api key"]
        if any(ind in err_msg for ind in api_key_indicators):
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired Gemini API key. Set GEMINI_API_KEY in backend/.env and restart the backend."
            )
        # Return 429 for quota errors
        if "429" in str(e) or "quota" in err_msg or "rate limit" in err_msg:
            raise HTTPException(
                status_code=429,
                detail="API quota exceeded. Wait a few minutes or check https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tools_registered": len(tool_registry.tools),
        "agent_state": agent.get_state()
    }

@app.get("/api/stats")
async def get_stats():
    """Get agent stats for dashboard"""
    return agent.get_stats()

@app.get("/api/tools")
async def list_tools():
    """List all registered tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_schema()
            }
            for tool in tool_registry.tools.values()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
