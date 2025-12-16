from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

from app.services.benchmark_service import BenchmarkService
from app.services.model_client import ModelClient
from app.services.analytics_engine import AnalyticsEngine

load_dotenv()

app = FastAPI(title="VAIA L5 - Model Comparison Platform")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_client = ModelClient(api_key=os.getenv("GEMINI_API_KEY"))
analytics_engine = AnalyticsEngine()
benchmark_service = BenchmarkService(model_client, analytics_engine)

# Request models
class BenchmarkRequest(BaseModel):
    prompts: List[str]
    models: List[str]
    repetitions: int = 3
    temperature: float = 0.7

class ComparisonRequest(BaseModel):
    model_names: List[str]

@app.get("/")
async def root():
    return {
        "service": "VAIA L5 - Model Comparison Platform",
        "status": "operational",
        "endpoints": {
            "models": "/api/models",
            "benchmark": "/api/benchmark",
            "compare": "/api/compare",
            "recommendations": "/api/recommendations"
        }
    }

@app.get("/api/models")
async def get_available_models():
    """Get list of available Gemini models with specifications."""
    return {
        "models": [
            {
                "id": "gemini-2.0-flash",
                "name": "Gemini 2.0 Flash",
                "parameters": "Unknown",
                "context_window": 1000000,
                "input_cost_per_1k": 0.0005,
                "output_cost_per_1k": 0.0015,
                "category": "efficient",
                "description": "Fast, cost-effective model for high-throughput tasks"
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "parameters": "Unknown",
                "context_window": 2000000,
                "input_cost_per_1k": 0.00125,
                "output_cost_per_1k": 0.005,
                "category": "balanced",
                "description": "Balanced performance for complex reasoning"
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "parameters": "Unknown",
                "context_window": 1000000,
                "input_cost_per_1k": 0.000075,
                "output_cost_per_1k": 0.0003,
                "category": "efficient",
                "description": "Optimized for speed and efficiency"
            }
        ]
    }

@app.post("/api/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """Execute benchmark across specified models and prompts."""
    try:
        results = await benchmark_service.run_benchmark(
            prompts=request.prompts,
            models=request.models,
            repetitions=request.repetitions,
            temperature=request.temperature
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare")
async def compare_models(request: ComparisonRequest):
    """Generate detailed comparison of specified models."""
    try:
        comparison = await analytics_engine.compare_models(request.model_names)
        return {"status": "success", "comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommendations")
async def get_recommendations(
    max_latency_ms: Optional[int] = None,
    max_cost_per_request: Optional[float] = None,
    min_quality_score: Optional[float] = None
):
    """Get model recommendations based on constraints."""
    try:
        recommendations = await analytics_engine.generate_recommendations(
            max_latency_ms=max_latency_ms,
            max_cost_per_request=max_cost_per_request,
            min_quality_score=min_quality_score
        )
        return {"status": "success", "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
