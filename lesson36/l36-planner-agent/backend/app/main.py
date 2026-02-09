"""FastAPI application for PlannerAgent service"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PlanRequest, PlanResponse
from .planner import PlannerAgent
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Metrics storage (in-memory, resets on restart)
metrics = {
    "total_queries": 0,
    "successful_decompositions": 0,
    "failed_decompositions": 0,
    "total_sub_queries": 0,
    "processing_times_ms": [],
    "decompositions_with_plan": 0,
}

app = FastAPI(
    title="L36 PlannerAgent Service",
    description="Query decomposition for Agentic RAG systems",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PlannerAgent
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

planner = PlannerAgent(api_key=GEMINI_API_KEY)

@app.get("/")
async def root():
    return {
        "service": "PlannerAgent",
        "lesson": "L36",
        "status": "operational",
        "capabilities": ["query_decomposition", "strategy_planning"]
    }

@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """Decompose query into sub-queries with retrieval strategy"""
    start_time = time.time()
    metrics["total_queries"] += 1
    
    try:
        plan = await planner.decompose_query(request)
        processing_time = (time.time() - start_time) * 1000
        
        metrics["successful_decompositions"] += 1
        metrics["total_sub_queries"] += len(plan.sub_queries)
        metrics["processing_times_ms"].append(processing_time)
        if len(metrics["processing_times_ms"]) > 100:
            metrics["processing_times_ms"] = metrics["processing_times_ms"][-100:]
        if plan.requires_decomposition and len(plan.sub_queries) > 1:
            metrics["decompositions_with_plan"] += 1
        
        return PlanResponse(
            success=True,
            plan=plan,
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        metrics["failed_decompositions"] += 1
        metrics["processing_times_ms"].append(processing_time)
        if len(metrics["processing_times_ms"]) > 100:
            metrics["processing_times_ms"] = metrics["processing_times_ms"][-100:]
        return PlanResponse(
            success=False,
            error=str(e),
            processing_time_ms=round(processing_time, 2)
        )


@app.get("/metrics")
async def get_metrics():
    """Return metrics for dashboard - updated by demo execution"""
    times = metrics["processing_times_ms"]
    avg_ms = sum(times) / len(times) if times else 0
    return {
        "total_queries": metrics["total_queries"],
        "successful_decompositions": metrics["successful_decompositions"],
        "failed_decompositions": metrics["failed_decompositions"],
        "total_sub_queries": metrics["total_sub_queries"],
        "decompositions_with_plan": metrics["decompositions_with_plan"],
        "avg_sub_queries": round(metrics["total_sub_queries"] / metrics["successful_decompositions"], 2) if metrics["successful_decompositions"] > 0 else 0,
        "avg_processing_ms": round(avg_ms, 2),
        "success_rate": round(100 * metrics["successful_decompositions"] / metrics["total_queries"], 1) if metrics["total_queries"] > 0 else 0,
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "planner-agent"}
