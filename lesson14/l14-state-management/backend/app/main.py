"""
FastAPI application with state management endpoints
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from services.state_manager import StateManager
from services.agent_engine import AgentEngine
from models.agent_state import AgentState

load_dotenv()

# Database connections
db_pool = None
redis_client = None
state_manager = None
agent_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    global db_pool, redis_client, state_manager, agent_engine
    
    # Create database pool
    db_pool = await asyncpg.create_pool(os.getenv('DATABASE_URL'))
    
    # Create Redis client
    redis_client = await redis.from_url(os.getenv('REDIS_URL'))
    
    # Initialize state manager
    state_manager = StateManager(db_pool, redis_client)
    await state_manager.initialize()
    
    # Initialize agent engine
    agent_engine = AgentEngine(state_manager, os.getenv('GEMINI_API_KEY'))
    
    yield
    
    # Cleanup
    await db_pool.close()
    await redis_client.close()

app = FastAPI(title="L14 State Management", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    session_id: str
    message: str

class StateRequest(BaseModel):
    session_id: str

class DiffRequest(BaseModel):
    session_id: str
    from_version: int
    to_version: int

class RollbackRequest(BaseModel):
    session_id: str
    target_version: int

@app.get("/")
async def root():
    return {"status": "L14 State Management Active", "version": "1.0"}

@app.post("/api/chat")
async def chat(request: MessageRequest):
    """Process chat message with state persistence"""
    try:
        result = await agent_engine.process_message(
            request.session_id,
            request.message
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/state/{session_id}")
async def get_state(session_id: str):
    """Retrieve current state for session"""
    state = await state_manager.load_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="State not found")
    return state.model_dump()

@app.post("/api/state/diff")
async def get_diff(request: DiffRequest):
    """Get difference between state versions"""
    diff = await state_manager.get_state_diff(
        request.session_id, 
        request.from_version, 
        request.to_version
    )
    if not diff:
        raise HTTPException(status_code=404, detail="Versions not found")
    return diff.model_dump()

@app.post("/api/state/rollback")
async def rollback(request: RollbackRequest):
    """Rollback state to previous version"""
    success = await state_manager.rollback_state(
        request.session_id, 
        request.target_version
    )
    if not success:
        raise HTTPException(status_code=400, detail="Rollback failed")
    return {"status": "success", "version": request.target_version}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if db_pool else "disconnected",
        "redis": "connected" if redis_client else "disconnected"
    }
