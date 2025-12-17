from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import structlog
from backend.gemini_client import GeminiAPIClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI(title="L6: LLM API Client", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client
client = GeminiAPIClient()

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class GenerateResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    request_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    wait_seconds: Optional[float] = None

@app.get("/")
async def root():
    return {
        "service": "L6: LLM API Client",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate content with production-grade API client."""
    result = await client.generate(
        prompt=request.prompt,
        generation_config={
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens
        }
    )
    return GenerateResponse(**result)

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive client statistics."""
    return client.get_stats()

@app.post("/api/stats/reset")
async def reset_stats():
    """Reset cost tracker statistics."""
    client.cost_tracker.reset_stats()
    return {"message": "Statistics reset successfully"}
