from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from .api import evaluation, benchmarks, metrics
from .core.database import init_db
from .services.websocket_manager import ws_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing RAG Evaluation System...")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="RAG Evaluation API",
    description="L27: RAG Evaluation Metrics & Benchmarks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["Benchmarks"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])

@app.get("/")
async def root():
    return {
        "lesson": "L27",
        "title": "RAG Evaluation Metrics & Benchmarks",
        "status": "operational"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
