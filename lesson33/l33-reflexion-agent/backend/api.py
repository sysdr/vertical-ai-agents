"""
FastAPI backend for Reflexion Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of cwd)
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / '.env'
load_dotenv(_env_path)

# Validate API key at import (before agent init)
_api_key = (os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or '').strip()
_PLACEHOLDERS = ('', 'your_gemini_api_key_here', 'your_api_key_here', 'your_key_here',
                 'your_gemini_api_key', 'replace_with_your_key')
_api_key_ok = bool(_api_key and _api_key.lower() not in _PLACEHOLDERS)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.reflexion_agent import ReflexionAgent
from backend.tools import DEFAULT_TOOLS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L33 Reflexion Agent API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=4)

class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None

class TaskResponse(BaseModel):
    success: bool
    result: str
    attempts: int
    reflections: List[Dict[str, Any]]
    session_id: str
    stats: Dict[str, Any]

@app.post("/api/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute task with reflexion"""
    try:
        logger.info(f"Received task: {request.task}")
        
        result = agent.run(
            task=request.task,
            session_id=request.session_id
        )
        
        stats = agent.get_reflection_stats()
        
        return TaskResponse(
            success=result['success'],
            result=result['result'],
            attempts=result['attempts'],
            reflections=result['reflections'],
            session_id=result['session_id'],
            stats=stats
        )
    
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": agent.model_name,
        "max_reflections": agent.max_reflections,
        "api_key_configured": _api_key_ok
    }

@app.get("/api/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": name,
                "description": tool.description
            }
            for name, tool in agent.tools.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
