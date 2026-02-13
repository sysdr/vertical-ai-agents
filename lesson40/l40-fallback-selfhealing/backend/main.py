from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fallback_orchestrator import FallbackOrchestrator
import os

app = FastAPI(title="L40 Fallback & Self-Healing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required. Set it before starting the backend.")
orchestrator = FallbackOrchestrator(GEMINI_API_KEY)

class QueryRequest(BaseModel):
    query: str
    simulate_failure: bool = False

@app.get("/")
async def root():
    return {
        "service": "L40 Fallback & Self-Healing",
        "status": "operational",
        "circuit_breaker": orchestrator.circuit_breaker.get_state()
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query with fallback logic"""
    try:
        response = await orchestrator.execute_with_fallback(
            request.query,
            simulate_failure=request.simulate_failure
        )
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get orchestrator metrics"""
    return orchestrator.get_metrics()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "circuit_breaker_state": orchestrator.circuit_breaker.state.value
    }
