from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import os
from dotenv import load_dotenv

from middleware.observability import ObservabilityMiddleware
from services.rag_service import InstrumentedRAGService
from services.metrics_collector import metrics
from pydantic import BaseModel
from pathlib import Path

# Load .env from backend directory (works when run from project root or backend/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await metrics.initialize()
    app.state.rag_service = InstrumentedRAGService(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    yield
    # Shutdown
    await metrics.shutdown()

app = FastAPI(title="L30: RAG Observability", lifespan=lifespan)

# Add observability middleware
app.add_middleware(ObservabilityMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def process_query(request: Request, query_req: QueryRequest):
    """Process RAG query with full instrumentation"""
    trace_id = request.state.trace_id
    
    result = await app.state.rag_service.process_query(
        query=query_req.query,
        trace_id=trace_id
    )
    
    return result

@app.get("/api/stats")
async def get_statistics(minutes: int = 60):
    """Get aggregate performance statistics"""
    stats = await metrics.get_stats(time_window_minutes=minutes)
    return stats

@app.get("/api/timeseries/{metric_name}")
async def get_timeseries(metric_name: str, minutes: int = 60):
    """Get time-series data for specific metric"""
    data = await metrics.get_timeseries(metric_name, minutes)
    return {"metric": metric_name, "data": data}

# WebSocket for real-time metrics streaming
active_connections = []

@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            stats = await metrics.get_stats(time_window_minutes=5)
            await websocket.send_json(stats.model_dump())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-observability"}
