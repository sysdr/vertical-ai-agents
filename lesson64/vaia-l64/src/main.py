"""
L64 – VAIA Agent: Containerised FastAPI application.
Provides /health/live, /health/ready, /metrics, and /v1/agent/infer endpoints.
Bridges from L63 (tested logic) → L65 (load tested serving).
"""
import asyncio
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    make_asgi_app, Counter, Histogram, Gauge, REGISTRY as PROM_REGISTRY
)
import chromadb

from src.core.config import Settings
from src.core.agent import VAIAAgent
from src.routers.infer import router as infer_router

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT    = Counter("vaia_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY  = Histogram("vaia_request_latency_seconds", "Latency", ["endpoint"],
                              buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5])
ACTIVE_REQUESTS  = Gauge("vaia_active_requests", "In-flight requests")
AGENT_READY      = Gauge("vaia_agent_ready", "1 if agent passed readiness check")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vaia.main")

settings = Settings()
agent: VAIAAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Starting VAIA agent warm-up…")
    agent = VAIAAgent(settings)
    await agent.warm_up()
    AGENT_READY.set(1)
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down agent.")
    AGENT_READY.set(0)


app = FastAPI(
    title="VAIA Agent – L64",
    description="Containerised VAIA agent. L64: CD & Containerization.",
    version="0.64.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus scrape endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Agent inference router
app.include_router(infer_router, prefix="/v1/agent")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start)
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        ACTIVE_REQUESTS.dec()
    return response


# ── Health endpoints ─────────────────────────────────────────────────────────

@app.get("/health/live", tags=["health"])
async def liveness():
    """Process-level health – never do expensive checks here."""
    return {"status": "ok", "version": "0.64.0"}


@app.get("/health/ready", tags=["health"])
async def readiness():
    """
    Dependency-level health: checks ChromaDB heartbeat.
    Returns 503 until agent fully warmed up (gates K8s traffic via readiness probe).
    """
    if agent is None or not agent.is_ready:
        raise HTTPException(status_code=503, detail="Agent not yet ready")
    try:
        chroma_ok = await asyncio.wait_for(agent.chroma_heartbeat(), timeout=2.0)
    except asyncio.TimeoutError:
        chroma_ok = False
    if not chroma_ok:
        AGENT_READY.set(0)
        raise HTTPException(status_code=503, detail="ChromaDB unreachable")
    AGENT_READY.set(1)
    return {"status": "ready"}


@app.get("/", tags=["root"])
async def root():
    return {
        "lesson": "L64 – CD & Containerization",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": {"live": "/health/live", "ready": "/health/ready"},
    }
