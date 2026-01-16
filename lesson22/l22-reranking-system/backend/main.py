from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import redis
import json
import hashlib
import time
import asyncio
from datetime import datetime
import numpy as np

app = FastAPI(title="L22 Reranking Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
GEMINI_API_KEY = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Redis cache
import os
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    CACHE_ENABLED = True
    print(f"✅ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    CACHE_ENABLED = False
    print(f"⚠️  Redis not available - caching disabled: {e}")

# Load cross-encoder model
print("📦 Loading cross-encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Model loaded")

# Metrics storage
metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_rerank_time": 0,
    "avg_improvement": 0,
    "rerank_times": []
}

class RerankRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]]
    top_k: int = 10
    use_cache: bool = True

class RerankResponse(BaseModel):
    query: str
    reranked_documents: List[Dict]
    original_scores: List[float]
    reranked_scores: List[float]
    improvement_pct: float
    latency_ms: float
    cache_hit: bool
    timestamp: str

@app.get("/")
async def root():
    return {
        "service": "L22 Reranking Service",
        "status": "operational",
        "cache_enabled": CACHE_ENABLED,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cache": "enabled" if CACHE_ENABLED else "disabled",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    start_time = time.time()
    metrics["total_requests"] += 1
    
    # Generate cache key
    doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(request.documents)]
    cache_key = f"rerank:{hashlib.md5(request.query.encode()).hexdigest()}:{hashlib.md5(','.join(doc_ids).encode()).hexdigest()}"
    
    # Check cache
    cache_hit = False
    if request.use_cache and CACHE_ENABLED:
        cached = redis_client.get(cache_key)
        if cached:
            cache_hit = True
            metrics["cache_hits"] += 1
            result = json.loads(cached)
            result["cache_hit"] = True
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result
    
    metrics["cache_misses"] += 1
    
    # Extract documents and original scores
    docs = [doc.get("text", str(doc)) for doc in request.documents]
    original_scores = [doc.get("score", 0.5) for doc in request.documents]
    
    # Rerank with cross-encoder
    query_doc_pairs = [[request.query, doc] for doc in docs]
    rerank_scores = cross_encoder.predict(query_doc_pairs).tolist()
    
    # Sort by reranked scores
    scored_docs = list(zip(request.documents, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k
    top_docs = scored_docs[:request.top_k]
    reranked_documents = [{"rank": i+1, **doc, "rerank_score": score} 
                          for i, (doc, score) in enumerate(top_docs)]
    reranked_scores = [score for _, score in top_docs]
    
    # Calculate improvement
    original_top_k = sorted(original_scores, reverse=True)[:request.top_k]
    improvement = ((np.mean(reranked_scores) - np.mean(original_top_k)) / 
                   max(np.mean(original_top_k), 0.01)) * 100
    
    latency = (time.time() - start_time) * 1000
    metrics["rerank_times"].append(latency)
    
    response = {
        "query": request.query,
        "reranked_documents": reranked_documents,
        "original_scores": original_scores[:request.top_k],
        "reranked_scores": reranked_scores,
        "improvement_pct": round(improvement, 2),
        "latency_ms": round(latency, 2),
        "cache_hit": cache_hit,
        "timestamp": datetime.now().isoformat()
    }
    
    # Cache result
    if CACHE_ENABLED:
        redis_client.setex(cache_key, 86400, json.dumps(response))
    
    return response

@app.post("/batch-rerank")
async def batch_rerank(queries: List[str], documents: List[Dict]):
    """Batch reranking for efficiency"""
    results = []
    for query in queries:
        request = RerankRequest(query=query, documents=documents)
        result = await rerank_documents(request)
        results.append(result)
    return {"batch_results": results, "count": len(results)}

@app.get("/metrics")
async def get_metrics():
    avg_time = np.mean(metrics["rerank_times"]) if metrics["rerank_times"] else 0
    cache_hit_rate = (metrics["cache_hits"] / max(metrics["total_requests"], 1)) * 100
    
    return {
        "total_requests": metrics["total_requests"],
        "cache_hit_rate": round(cache_hit_rate, 2),
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
        "avg_latency_ms": round(avg_time, 2),
        "p95_latency_ms": round(np.percentile(metrics["rerank_times"], 95), 2) if metrics["rerank_times"] else 0,
        "p99_latency_ms": round(np.percentile(metrics["rerank_times"], 99), 2) if metrics["rerank_times"] else 0
    }

class CompareRequest(BaseModel):
    query: str
    documents: List[Dict[str, Any]]

@app.post("/compare")
async def compare_retrieval_methods(request: CompareRequest):
    """Compare bi-encoder vs cross-encoder results"""
    
    query = request.query
    documents = request.documents
    
    # Simulate bi-encoder scores (from L21)
    bi_encoder_scores = [doc.get("score", np.random.uniform(0.3, 0.9)) 
                         for doc in documents]
    bi_encoder_ranked = sorted(zip(documents, bi_encoder_scores), 
                               key=lambda x: x[1], reverse=True)
    
    # Cross-encoder reranking
    docs = [doc.get("text", str(doc)) for doc in documents]
    query_doc_pairs = [[query, doc] for doc in docs]
    cross_encoder_scores = cross_encoder.predict(query_doc_pairs).tolist()
    cross_encoder_ranked = sorted(zip(documents, cross_encoder_scores), 
                                  key=lambda x: x[1], reverse=True)
    
    return {
        "query": query,
        "bi_encoder_top_5": [
            {"rank": i+1, "text": doc.get("text", "")[:100], "score": score}
            for i, (doc, score) in enumerate(bi_encoder_ranked[:5])
        ],
        "cross_encoder_top_5": [
            {"rank": i+1, "text": doc.get("text", "")[:100], "score": score}
            for i, (doc, score) in enumerate(cross_encoder_ranked[:5])
        ],
        "ranking_changes": sum(1 for i in range(min(5, len(documents)))
                              if bi_encoder_ranked[i][0] != cross_encoder_ranked[i][0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
