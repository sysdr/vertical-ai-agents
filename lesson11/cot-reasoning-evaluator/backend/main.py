"""FastAPI backend for CoT Reasoning Evaluator"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from cot_agent import CoTAgent

app = FastAPI(title="CoT Reasoning Evaluator API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CoT agent
try:
    agent = CoTAgent(data_dir=Path("/app/data") if Path("/app").exists() else Path("../data"))
except ValueError as e:
    # API key not set
    agent = None
    print(f"ERROR: {e}")
    print("Please set GEMINI_API_KEY environment variable before starting the server.")

class QueryRequest(BaseModel):
    query: str
    style: str = "standard"

class MemoryResponse(BaseModel):
    traces: List[dict]
    total_count: int

@app.get("/")
async def root():
    return {
        "service": "CoT Reasoning Evaluator",
        "lesson": "L11",
        "status": "operational"
    }

@app.post("/api/reason")
async def reason_with_cot(request: QueryRequest):
    """Process query with Chain-of-Thought reasoning"""
    if agent is None:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is not set. Please set it and restart the server."
        )
    try:
        result = await agent.reason_with_cot(request.query, request.style)
        return result
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages for common issues
        if "quota" in error_msg.lower() or "429" in error_msg:
            error_msg = f"API Quota Exceeded: {error_msg}\n\nPlease wait a few minutes and try again, or check your Google Cloud Console for quota limits."
        elif "403" in error_msg or "api key" in error_msg.lower():
            error_msg = f"API Key Error: {error_msg}\n\nPlease verify your GEMINI_API_KEY is correct and has not been revoked."
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/memory")
async def get_memory(limit: int = 10):
    """Retrieve recent reasoning traces"""
    if agent is None:
        return MemoryResponse(traces=[], total_count=0)
    traces = agent.get_memory(limit)
    return MemoryResponse(traces=traces, total_count=len(agent.memory))

@app.get("/api/quality-traces")
async def get_quality_traces(min_score: float = 0.7):
    """Get high-quality reasoning traces for few-shot examples"""
    if agent is None:
        return {"traces": [], "count": 0}
    traces = agent.get_high_quality_traces(min_score)
    return {"traces": traces, "count": len(traces)}

@app.get("/api/stats")
async def get_statistics():
    """Get reasoning quality statistics"""
    if agent is None or not agent.memory:
        return {"message": "No reasoning traces yet"}
    
    scores = [trace["quality_scores"]["overall_quality"] for trace in agent.memory]
    return {
        "total_traces": len(agent.memory),
        "avg_quality": sum(scores) / len(scores),
        "max_quality": max(scores),
        "min_quality": min(scores),
        "high_quality_count": len([s for s in scores if s >= 0.7])
    }

@app.delete("/api/memory")
async def clear_memory():
    """Clear all stored reasoning traces"""
    if agent is None:
        return {"message": "Agent not initialized"}
    agent.memory = []
    agent._save_memory()
    return {"message": "Memory cleared"}
