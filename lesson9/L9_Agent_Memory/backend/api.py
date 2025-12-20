"""
FastAPI Memory API
Endpoints for storing and retrieving agent memory
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from memory_manager import MemoryManager

app = FastAPI(title="Agent Memory API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory manager
memory = MemoryManager()

# Request models
class MessageStore(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: Optional[dict] = None

class LongTermStore(BaseModel):
    user_id: str
    fact: dict

class SearchRequest(BaseModel):
    user_id: str
    keywords: Optional[List[str]] = None
    limit: int = 5

@app.post("/api/store")
async def store_message(request: MessageStore):
    """Store message in short-term memory"""
    message = memory.store_message(
        request.session_id,
        request.role,
        request.content,
        request.metadata
    )
    return {"success": True, "message": message}

@app.get("/api/recall/{session_id}")
async def recall_session(session_id: str, limit: int = 10):
    """Recall messages from short-term memory"""
    messages = memory.recall_session(session_id, limit)
    return {"success": True, "messages": messages, "count": len(messages)}

@app.post("/api/longterm/store")
async def store_longterm(request: LongTermStore):
    """Store fact in long-term memory"""
    success = memory.save_to_longterm(request.user_id, request.fact)
    if success:
        return {"success": True, "message": "Fact stored in long-term memory"}
    raise HTTPException(status_code=500, detail="Failed to store fact")

@app.post("/api/longterm/search")
async def search_longterm(request: SearchRequest):
    """Search long-term memory"""
    results = memory.search_longterm(
        request.user_id,
        request.keywords,
        request.limit
    )
    return {"success": True, "results": results, "count": len(results)}

@app.get("/api/metrics")
async def get_metrics():
    """Get memory system metrics"""
    metrics = memory.get_metrics()
    return {"success": True, "metrics": metrics}

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear short-term memory for session"""
    success = memory.clear_session(session_id)
    if success:
        return {"success": True, "message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Agent Memory API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
