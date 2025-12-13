from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
import statistics

app = FastAPI(title="LLM Parameter Analysis Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8")

# Model specifications database (May 2025)
MODEL_SPECS = {
    "gemini-2.0-flash-exp": {
        "parameters": "7B",
        "context_window": 1000000,
        "input_price_per_1m": 0.075,
        "output_price_per_1m": 0.30,
        "cached_input_price_per_1m": 0.01875,
        "latency_estimate_ms": 85,
        "max_rpm": 1500,
        "description": "Ultra-fast inference for high-volume tasks"
    },
    "gemini-1.5-pro": {
        "parameters": "70B",
        "context_window": 2000000,
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 5.00,
        "cached_input_price_per_1m": 0.3125,
        "latency_estimate_ms": 420,
        "max_rpm": 360,
        "description": "Balanced performance for production workloads"
    },
    "gemini-1.5-flash": {
        "parameters": "7B",
        "context_window": 1000000,
        "input_price_per_1m": 0.075,
        "output_price_per_1m": 0.30,
        "cached_input_price_per_1m": 0.01875,
        "latency_estimate_ms": 95,
        "max_rpm": 1500,
        "description": "Cost-effective for classification and extraction"
    }
}

class CostCalculationRequest(BaseModel):
    model_name: str
    requests_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    cache_hit_rate: float = 0.5

class PerformanceTestRequest(BaseModel):
    model_name: str
    test_prompt: str
    num_iterations: int = 5

class ContextAnalysisRequest(BaseModel):
    model_name: str
    sample_texts: List[str]

@app.get("/")
async def root():
    return {
        "service": "LLM Parameter Analysis Platform",
        "version": "1.0.0",
        "lesson": "L4",
        "status": "operational"
    }

@app.get("/models")
async def get_models():
    """Retrieve all available model specifications"""
    return {
        "models": MODEL_SPECS,
        "count": len(MODEL_SPECS),
        "last_updated": "2025-05-01"
    }

@app.get("/models/{model_name}")
async def get_model_specs(model_name: str):
    """Get detailed specifications for a specific model"""
    if model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    specs = MODEL_SPECS[model_name]
    
    # Calculate additional metrics
    params_numeric = float(specs["parameters"].replace("B", ""))
    compute_intensity = params_numeric * specs["context_window"] / 1_000_000
    
    return {
        **specs,
        "compute_intensity_score": round(compute_intensity, 2),
        "cost_efficiency_score": round(
            1000 / (specs["input_price_per_1m"] + specs["output_price_per_1m"]), 2
        ),
        "speed_score": round(1000 / specs["latency_estimate_ms"], 2)
    }

@app.post("/analyze/cost")
async def calculate_cost(request: CostCalculationRequest):
    """Calculate projected monthly costs with caching benefits"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = MODEL_SPECS[request.model_name]
    
    # Calculate effective input tokens with caching
    cache_savings_factor = 1 - (request.cache_hit_rate * 0.75)
    effective_input_tokens = request.avg_input_tokens * cache_savings_factor
    
    # Daily costs
    cached_input_cost = (request.avg_input_tokens * request.cache_hit_rate * 
                        model["cached_input_price_per_1m"] / 1_000_000)
    uncached_input_cost = (request.avg_input_tokens * (1 - request.cache_hit_rate) * 
                          model["input_price_per_1m"] / 1_000_000)
    output_cost = request.avg_output_tokens * model["output_price_per_1m"] / 1_000_000
    
    cost_per_request = cached_input_cost + uncached_input_cost + output_cost
    daily_cost = cost_per_request * request.requests_per_day
    
    # Without caching
    cost_without_cache = (
        (request.avg_input_tokens * model["input_price_per_1m"] / 1_000_000) +
        (request.avg_output_tokens * model["output_price_per_1m"] / 1_000_000)
    ) * request.requests_per_day
    
    cache_savings = cost_without_cache - daily_cost
    
    return {
        "model": request.model_name,
        "cost_per_request": round(cost_per_request, 6),
        "daily_cost": round(daily_cost, 2),
        "monthly_cost": round(daily_cost * 30, 2),
        "yearly_cost": round(daily_cost * 365, 2),
        "cache_savings_daily": round(cache_savings, 2),
        "cache_savings_monthly": round(cache_savings * 30, 2),
        "total_requests_monthly": request.requests_per_day * 30,
        "breakdown": {
            "cached_input_cost_per_request": round(cached_input_cost, 8),
            "uncached_input_cost_per_request": round(uncached_input_cost, 8),
            "output_cost_per_request": round(output_cost, 8)
        }
    }

@app.post("/analyze/performance")
async def test_performance(request: PerformanceTestRequest):
    """Run actual inference tests to measure performance"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = genai.GenerativeModel(request.model_name)
        latencies = []
        token_counts = []
        
        for i in range(request.num_iterations):
            start_time = datetime.now()
            
            try:
                response = model.generate_content(
                    request.test_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=100,
                        temperature=0.7,
                    )
                )
            except Exception as api_error:
                error_str = str(api_error)
                # Check for quota/rate limit errors
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    raise HTTPException(
                        status_code=429,
                        detail=f"API Quota Exceeded: {error_str}. The free tier quota for Gemini API has been reached. Please wait for quota reset or upgrade your plan."
                    )
                raise
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            latencies.append(latency)
            
            # Count tokens
            try:
                token_count = model.count_tokens(request.test_prompt)
                token_counts.append(token_count.total_tokens)
            except Exception as token_error:
                # If token counting fails, estimate from prompt length
                estimated_tokens = len(request.test_prompt.split()) * 1.3
                token_counts.append(int(estimated_tokens))
            
            await asyncio.sleep(0.5)  # Rate limit compliance
        
        return {
            "model": request.model_name,
            "iterations": request.num_iterations,
            "latency_ms": {
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) > 2 else round(max(latencies), 2)
            },
            "input_tokens": {
                "mean": round(statistics.mean(token_counts), 2),
                "max": max(token_counts)
            },
            "estimated_spec": MODEL_SPECS[request.model_name]["latency_estimate_ms"],
            "variance_from_spec": round(
                statistics.mean(latencies) - MODEL_SPECS[request.model_name]["latency_estimate_ms"], 2
            )
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 429) as-is
        raise
    except Exception as e:
        error_str = str(e)
        # Check if it's a quota error that wasn't caught earlier
        if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
            raise HTTPException(
                status_code=429,
                detail=f"API Quota Exceeded: {error_str}. The free tier quota for Gemini API has been reached."
            )
        raise HTTPException(status_code=500, detail=f"Performance test failed: {error_str}")

@app.post("/analyze/context")
async def analyze_context_usage(request: ContextAnalysisRequest):
    """Analyze context window efficiency from sample texts"""
    
    if request.model_name not in MODEL_SPECS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = genai.GenerativeModel(request.model_name)
        token_counts = []
        
        for text in request.sample_texts:
            count = model.count_tokens(text)
            token_counts.append(count.total_tokens)
        
        if not token_counts:
            raise HTTPException(status_code=400, detail="No valid texts provided")
        
        sorted_counts = sorted(token_counts)
        p50_idx = len(sorted_counts) // 2
        p95_idx = int(len(sorted_counts) * 0.95)
        p99_idx = int(len(sorted_counts) * 0.99)
        
        p50 = sorted_counts[p50_idx]
        p95 = sorted_counts[p95_idx] if p95_idx < len(sorted_counts) else sorted_counts[-1]
        p99 = sorted_counts[p99_idx] if p99_idx < len(sorted_counts) else sorted_counts[-1]
        
        context_limit = MODEL_SPECS[request.model_name]["context_window"]
        recommended_limit = p99
        
        # Calculate potential savings
        avg_tokens = statistics.mean(token_counts)
        potential_savings_pct = ((context_limit - recommended_limit) / context_limit) * 100
        
        return {
            "model": request.model_name,
            "samples_analyzed": len(token_counts),
            "token_distribution": {
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": round(avg_tokens, 2),
                "median": p50,
                "p95": p95,
                "p99": p99
            },
            "context_window_max": context_limit,
            "recommended_context_limit": recommended_limit,
            "efficiency_metrics": {
                "utilization_rate": round((recommended_limit / context_limit) * 100, 2),
                "potential_savings_pct": round(potential_savings_pct, 2),
                "tokens_saved_per_request": context_limit - recommended_limit
            },
            "recommendations": [
                f"Set context limit to {recommended_limit} tokens (covers 99% of requests)",
                f"Potential cost reduction: {round(potential_savings_pct, 1)}%",
                "Implement context compression for outliers above p99"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")

@app.post("/compare")
async def compare_models(
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    cache_hit_rate: float = 0.5
):
    """Compare all models for given usage pattern"""
    
    comparisons = []
    
    for model_name, specs in MODEL_SPECS.items():
        # Calculate costs
        cache_savings_factor = 1 - (cache_hit_rate * 0.75)
        
        cached_input_cost = (avg_input_tokens * cache_hit_rate * 
                            specs["cached_input_price_per_1m"] / 1_000_000)
        uncached_input_cost = (avg_input_tokens * (1 - cache_hit_rate) * 
                              specs["input_price_per_1m"] / 1_000_000)
        output_cost = avg_output_tokens * specs["output_price_per_1m"] / 1_000_000
        
        cost_per_request = cached_input_cost + uncached_input_cost + output_cost
        monthly_cost = cost_per_request * requests_per_day * 30
        
        comparisons.append({
            "model": model_name,
            "parameters": specs["parameters"],
            "monthly_cost": round(monthly_cost, 2),
            "cost_per_request": round(cost_per_request, 6),
            "latency_ms": specs["latency_estimate_ms"],
            "context_window": specs["context_window"],
            "max_rpm": specs["max_rpm"],
            "cost_efficiency_score": round(1000 / monthly_cost, 4),
            "speed_score": round(1000 / specs["latency_estimate_ms"], 2)
        })
    
    # Sort by cost efficiency
    comparisons.sort(key=lambda x: x["cost_efficiency_score"], reverse=True)
    
    return {
        "usage_pattern": {
            "requests_per_day": requests_per_day,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "cache_hit_rate": cache_hit_rate
        },
        "comparisons": comparisons,
        "recommendation": comparisons[0]["model"],
        "cost_range": {
            "min": min(c["monthly_cost"] for c in comparisons),
            "max": max(c["monthly_cost"] for c in comparisons)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
