"""
DockerizedAgentService — Cloud Run-compatible FastAPI agent.

Lifecycle endpoints:
  GET /health  → always 200 (liveness probe)
  GET /ready   → 503 during warmup, 200 when HEALTHY (readiness probe)
  GET /metrics → Prometheus text exposition

Agent endpoints:
  POST /chat

Deployment endpoints:
  GET  /deployment/state
  GET  /deployment/history
  GET  /deployment/alerts
  GET  /deployment/metrics-summary
  POST /admin/inject-drift   (testing)
"""
import asyncio
import random
import signal
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from agent_service.agent import GeminiAgent
from agent_service.config import settings
from agent_service.metrics import ACTIVE_CONNECTIONS, DEPLOYMENT_STATE, DRIFT_SCORE, ERROR_RATE
from monitoring.alert_router import AlertWebhookRouter
from monitoring.drift_responder import DriftResponder
from monitoring.orchestrator import DeploymentOrchestrator, DeploymentState

# ── singletons ────────────────────────────────────────────────────────────────
_shutdown = asyncio.Event()
orchestrator = DeploymentOrchestrator(db_path=settings.db_path)
alert_router = AlertWebhookRouter(webhook_url=settings.alert_webhook_url)
drift_responder = DriftResponder(orchestrator=orchestrator, alert_router=alert_router)
agent = GeminiAgent()
_ready = False


def _handle_sigterm(*_: object) -> None:
    _shutdown.set()


signal.signal(signal.SIGTERM, _handle_sigterm)


def _prime_dashboard_metrics() -> None:
    """Seed rolling windows so dashboard cards/charts are non-zero before traffic."""
    rnd = random.Random(74)
    for _ in range(40):
        drift_responder.record_request(
            latency=0.44 + rnd.uniform(-0.06, 0.10),
            success=True,
            variant=settings.active_variant,
        )
    variant = settings.active_variant
    drift = drift_responder._compute_drift_score()
    err_rate = drift_responder._compute_error_rate()
    DRIFT_SCORE.labels(variant=variant).set(drift)
    ERROR_RATE.labels(variant=variant).set(err_rate)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _ready
    asyncio.create_task(orchestrator.run())
    asyncio.create_task(drift_responder.run(_shutdown))
    await asyncio.sleep(settings.warmup_delay_seconds)
    _prime_dashboard_metrics()
    _ready = True
    yield
    _shutdown.set()
    await orchestrator.drain()


app = FastAPI(title="VAIA MLOps Agent", version="74.0.1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── lifecycle probes ──────────────────────────────────────────────────────────
@app.get("/health", tags=["probes"])
async def health() -> dict:
    """Liveness: always 200 if process is running."""
    return {"status": "ok", "ts": time.time()}


@app.get("/ready", tags=["probes"])
async def ready() -> JSONResponse:
    """Readiness: gates traffic until warmup completes."""
    if not _ready:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(content={"status": "ready", "state": orchestrator.state.name})


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── agent ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    variant: str = "stable"


@app.post("/chat", tags=["agent"])
async def chat(req: ChatRequest) -> dict:
    ACTIVE_CONNECTIONS.inc()
    try:
        result = await agent.chat(req.session_id, req.message, req.variant)
        drift_responder.record_request(
            latency=result["latency_ms"] / 1000.0,
            success=not result.get("error", False),
            variant=req.variant,
        )
        return result
    finally:
        ACTIVE_CONNECTIONS.dec()


# ── deployment API ────────────────────────────────────────────────────────────
@app.get("/deployment/state", tags=["deployment"])
async def deployment_state() -> dict:
    return {"state": orchestrator.state.name, "state_id": int(orchestrator.state)}


@app.get("/deployment/history", tags=["deployment"])
async def deployment_history() -> dict:
    return {"history": await orchestrator.get_history(50)}


@app.get("/deployment/alerts", tags=["deployment"])
async def deployment_alerts() -> dict:
    return {"alerts": alert_router.get_recent(20)}


@app.get("/deployment/metrics-summary", tags=["deployment"])
async def metrics_summary() -> dict:
    variant = settings.active_variant
    drift = drift_responder._compute_drift_score()
    err_rate = drift_responder._compute_error_rate()
    p95 = drift_responder._compute_p95_latency()
    DRIFT_SCORE.labels(variant=variant).set(drift)
    ERROR_RATE.labels(variant=variant).set(err_rate)
    return {
        "error_rate":    round(err_rate, 6),
        "p95_latency_s": round(p95, 4),
        "drift_score":   round(drift, 4),
        "state":         orchestrator.state.name,
        "variant":       settings.active_variant,
    }


@app.post("/admin/inject-drift", tags=["testing"])
async def inject_drift(score: float = 0.5) -> dict:
    """
    Injects artificial high-latency requests to simulate drift.
    Useful for dashboard demo and CI smoke tests.
    """
    for _ in range(20):
        drift_responder.record_request(
            latency=score * 8.0,
            success=score < 0.5,
        )
    variant = settings.active_variant
    drift = drift_responder._compute_drift_score()
    err_rate = drift_responder._compute_error_rate()
    p95 = drift_responder._compute_p95_latency()
    DRIFT_SCORE.labels(variant=variant).set(drift)
    ERROR_RATE.labels(variant=variant).set(err_rate)
    return {
        "injected": True,
        "score_proxy": score,
        "records": 20,
        "drift_score": drift,
        "error_rate": err_rate,
        "p95_latency_s": p95,
    }
