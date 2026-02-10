from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import asyncio
import logging
from datetime import datetime
import json

from retriever_agent import RetrieverAgent
from reranker_tool import GeminiReranker
from vector_store import VectorStore
from planner_agent import PlannerAgent
from metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components (at module level for lifespan)
vector_store = VectorStore()
reranker = GeminiReranker()
planner = PlannerAgent()
retriever = RetrieverAgent(vector_store, reranker)
metrics = MetricsCollector()

# WebSocket connections
active_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize vector store. Shutdown: no-op."""
    logger.info("Initializing vector store...")
    await vector_store.initialize()
    logger.info("Vector store initialized")
    yield
    # shutdown if needed later

app = FastAPI(title="L37 RetrieverAgent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranking: bool = True
    use_hybrid: bool = False

class RetrievalResponse(BaseModel):
    chunks: List[Dict]
    metrics: Dict
    sub_queries: List[str]
    execution_time_ms: float

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: QueryRequest):
    """Execute complete retrieval pipeline: plan -> search -> rerank"""
    start_time = datetime.now()
    
    try:
        # Stage 1: Query planning
        logger.info(f"Planning query: {request.query}")
        sub_queries = await planner.decompose(request.query)
        
        await broadcast_event({
            "type": "planning_complete",
            "sub_queries": sub_queries,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stage 2: Retrieval with optional reranking
        logger.info(f"Retrieving with {len(sub_queries)} sub-queries")
        chunks = await retriever.retrieve(
            original_query=request.query,
            sub_queries=sub_queries,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_hybrid=request.use_hybrid
        )
        
        await broadcast_event({
            "type": "retrieval_complete",
            "chunks_count": len(chunks),
            "timestamp": datetime.now().isoformat()
        })
        
        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        retrieval_metrics = {
            "total_chunks": len(chunks),
            "execution_time_ms": execution_time,
            "sub_queries_count": len(sub_queries),
            "reranking_enabled": request.use_reranking,
            "hybrid_enabled": request.use_hybrid,
            "avg_vector_score": sum(c.get("vector_score", 0) for c in chunks) / len(chunks) if chunks else 0,
            "avg_rerank_score": sum(c.get("rerank_score", 0) for c in chunks) / len(chunks) if chunks and request.use_reranking else None
        }
        
        metrics.record_retrieval(retrieval_metrics)
        
        return RetrievalResponse(
            chunks=chunks,
            metrics=retrieval_metrics,
            sub_queries=sub_queries,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    return metrics.get_summary()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_event(event: Dict):
    """Broadcast event to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_json(event)
        except:
            pass

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
