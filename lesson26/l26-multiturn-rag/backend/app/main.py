from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv
from pathlib import Path
from app.api import routes
from app.services.redis_service import redis_service
from app.services.rag_service import rag_service

# Load environment variables from .env file - try multiple possible locations
env_paths = [
    Path(__file__).parent.parent / ".env",  # backend/.env
    Path.cwd() / "backend" / ".env",  # if running from project root
    Path.cwd() / ".env",  # if running from backend directory
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    # Fallback: try loading from current directory
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Multi-Turn RAG System...")
    await redis_service.connect()
    await rag_service.initialize()
    logger.info("System ready")
    yield
    logger.info("Shutting down...")
    await redis_service.disconnect()

app = FastAPI(
    title="L26: Multi-Turn RAG System",
    description="Conversational RAG with ambiguity handling",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")

@app.get("/")
async def root():
    return {
        "service": "Multi-Turn RAG System",
        "lesson": "L26",
        "status": "running"
    }

@app.get("/health")
async def health():
    redis_status = await redis_service.health_check()
    rag_status = rag_service.health_check()
    
    return {
        "status": "healthy" if redis_status and rag_status else "degraded",
        "redis": redis_status,
        "rag": rag_status
    }
