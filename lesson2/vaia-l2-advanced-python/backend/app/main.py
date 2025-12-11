from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
import os
import time
from functools import wraps
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VAIA L2: Advanced Python Patterns")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models - Type-Safe Data Contracts
# ============================================================================

class AgentRequest(BaseModel):
    """Type-safe request model with validation"""
    agent_id: str = Field(..., pattern=r'^agent-[a-z0-9]{8}$')
    prompts: List[str] = Field(..., min_items=1, max_items=100)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    
    @field_validator('prompts')
    def validate_prompts(cls, v):
        if any(len(p) > 5000 for p in v):
            raise ValueError("Individual prompts must be under 5000 characters")
        return v

class AgentResponse(BaseModel):
    """Type-safe response model"""
    agent_id: str
    results: List[str]
    latency_ms: float
    timestamp: str
    cached: bool = False

class SystemMetrics(BaseModel):
    """System performance metrics"""
    total_requests: int
    active_tasks: int
    avg_latency_ms: float
    success_rate: float
    cache_hit_rate: float
    timestamp: str

# ============================================================================
# Production Decorators
# ============================================================================

# Global metrics
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "cache_hits": 0,
    "cache_misses": 0
}

# Simple in-memory cache
response_cache: Dict[str, Any] = {}

def with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator: Automatic retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Max retries exceeded: {str(e)}")
                        raise
                    wait_time = backoff ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator

def with_cache(ttl_seconds: int = 60):
    """Decorator: Simple caching layer"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function args
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            if cache_key in response_cache:
                cached_data, cached_time = response_cache[cache_key]
                if time.time() - cached_time < ttl_seconds:
                    metrics["cache_hits"] += 1
                    logger.info(f"Cache hit: {cache_key[:50]}")
                    return cached_data, True  # Return data and cache status
            
            # Cache miss - execute function
            metrics["cache_misses"] += 1
            result = await func(*args, **kwargs)
            response_cache[cache_key] = (result, time.time())
            return result, False  # Return data and cache status
        return wrapper
    return decorator

def with_performance_tracking(func):
    """Decorator: Track execution time and success rate"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        metrics["total_requests"] += 1
        
        try:
            result = await func(*args, **kwargs)
            metrics["successful_requests"] += 1
            return result
        except Exception as e:
            metrics["failed_requests"] += 1
            raise
        finally:
            latency = (time.time() - start_time) * 1000
            metrics["total_latency"] += latency
            logger.info(f"Request completed in {latency:.2f}ms")
    
    return wrapper

# ============================================================================
# Async Business Logic
# ============================================================================

GEMINI_API_KEY = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
# Allows running without hitting the real API when quota is exhausted
GEMINI_USE_FALLBACK = os.getenv("GEMINI_USE_FALLBACK", "1") == "1"

@with_retry(max_attempts=3, backoff=2.0)
async def call_gemini_api(session: aiohttp.ClientSession, prompt: str, temperature: float, max_tokens: int) -> str:
    """Async Gemini API call with retry logic"""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        async with session.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_text = await response.text()

                # Fallback path for quota exhaustion or rate limits to keep the demo usable
                if response.status == 429 and GEMINI_USE_FALLBACK:
                    logger.warning("Gemini quota exhausted; returning fallback demo response.")
                    return f"[DEMO] Processed (fallback) → {prompt[:200]}"

                raise HTTPException(status_code=response.status, detail=f"Gemini API error: {error_text}")
            
            data = await response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    except asyncio.TimeoutError:
        if GEMINI_USE_FALLBACK:
            logger.warning("Gemini timeout; returning fallback demo response.")
            return f"[DEMO] Timeout fallback → {prompt[:200]}"
        raise HTTPException(status_code=504, detail="Gemini API timeout")
    except Exception as e:
        # Provide a resilient demo-mode fallback for any upstream error if enabled
        if GEMINI_USE_FALLBACK:
            logger.warning(f"Gemini error '{e}'; returning fallback demo response.")
            return f"[DEMO] Error fallback → {prompt[:200]}"
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {str(e)}")

@with_cache(ttl_seconds=300)
@with_performance_tracking
async def process_prompts_async(prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
    """Process multiple prompts concurrently with caching"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_gemini_api(session, prompt, temperature, max_tokens)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Prompt {i} failed: {str(result)}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/process", response_model=AgentResponse)
async def process_agent_request(request: AgentRequest):
    """
    Process AI agent requests asynchronously with validation
    Demonstrates: async/await, Pydantic validation, decorators
    """
    start_time = time.time()
    
    try:
        # Process prompts with caching and retry logic
        results, cached = await process_prompts_async(
            request.prompts,
            request.temperature,
            request.max_tokens
        )
        
        latency = (time.time() - start_time) * 1000
        
        return AgentResponse(
            agent_id=request.agent_id,
            results=results,
            latency_ms=round(latency, 2),
            timestamp=datetime.now().isoformat(),
            cached=cached
        )
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get real-time system performance metrics"""
    total_reqs = metrics["total_requests"]
    successful = metrics["successful_requests"]
    
    return SystemMetrics(
        total_requests=total_reqs,
        active_tasks=len(asyncio.all_tasks()),
        avg_latency_ms=round(metrics["total_latency"] / max(total_reqs, 1), 2),
        success_rate=round(successful / max(total_reqs, 1), 3),
        cache_hit_rate=round(
            metrics["cache_hits"] / max(metrics["cache_hits"] + metrics["cache_misses"], 1),
            3
        ),
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "async_enabled": True,
        "decorators_active": True,
        "validation_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
