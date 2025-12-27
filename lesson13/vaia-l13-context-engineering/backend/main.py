from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from api.routes import context_routes
from utils.token_counter import TokenCounter
from services.summarizer import SummarizerService
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Download NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")
    
    yield
    
    logger.info("Shutting down Context Engineering service")

app = FastAPI(
    title="VAIA L13: Context Engineering",
    description="Production-grade context management and optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(context_routes.router, prefix="/api/v1", tags=["context"])

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "context-engineering",
        "lesson": "L13"
    }

@app.get("/")
async def root():
    return {
        "message": "VAIA L13: Context Engineering API",
        "endpoints": [
            "/api/v1/count-tokens",
            "/api/v1/summarize",
            "/api/v1/optimize-context",
            "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
