from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.orchestrator import RAGOrchestrator
import asyncio
import json
import os

app = FastAPI(title="L35 Agentic RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - API key from env (set GEMINI_API_KEY)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
orchestrator = RAGOrchestrator(GEMINI_API_KEY, max_tokens=3000, max_iterations=5)

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 3000

@app.get("/")
async def root():
    return {
        "service": "L35 Agentic RAG System",
        "version": "1.0.0",
        "agents": ["Planner", "Retriever", "Validator", "Synthesizer"]
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query through agentic RAG pipeline"""
    # Create new orchestrator for this query
    orch = RAGOrchestrator(GEMINI_API_KEY, max_tokens=request.max_tokens)
    result = await orch.process_query(request.query)
    return result.to_dict()

@app.get("/health")
async def health():
    return {"status": "healthy", "agents": 4}

# WebSocket for real-time updates
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # Create orchestrator with callback
            orch = RAGOrchestrator(
                GEMINI_API_KEY, 
                max_tokens=request.get("max_tokens", 3000)
            )
            
            async def send_update(state):
                await websocket.send_json(state)
            
            orch.register_callback(send_update)
            
            # Process query
            result = await orch.process_query(request["query"])
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
