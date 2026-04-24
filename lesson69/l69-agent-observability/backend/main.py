"""
L69 VAIA: FastAPI observability service.
REST + WebSocket access to agent traces.
Writes metric_snapshots consumed by L70 alerting.
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Set

import aiosqlite
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ObservableAgent
from db import get_traces, get_trace, get_metrics
from tracer import init_db, DB_PATH

# ── WebSocket hub ─────────────────────────────────────────────────────────────

_ws_clients: Set[WebSocket] = set()


async def broadcast(payload: dict) -> None:
    dead: Set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db(DB_PATH)
    print(f"[L69] Observability service listening on :{os.environ.get('PORT', 8069)}")
    yield
    print("[L69] Shutdown complete")


app = FastAPI(title="VAIA L69 – Agent Observability & Logging", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str
    context: Optional[dict] = None


class AgentResponse(BaseModel):
    trace_id: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/agent/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest):
    agent = ObservableAgent()
    loop  = asyncio.get_event_loop()

    result = await loop.run_in_executor(None, agent.run, req.query, req.context)

    # Allow BatchSpanProcessor to flush before we query for trace_id
    await asyncio.sleep(0.65)
    traces   = get_traces(limit=1)
    trace_id = traces[0]["trace_id"] if traces else "unknown"

    # Write metric_snapshots for L70 alerting
    async with aiosqlite.connect(DB_PATH) as db:
        lat_breached  = 1 if result["latency_ms"] > 5000.0 else 0
        cost_breached = 1 if result["cost_usd"]   > 0.01   else 0
        await db.execute("""
            INSERT INTO metric_snapshots (metric, value, threshold, breached, trace_id)
            VALUES ('latency_ms', ?, 5000.0, ?, ?)
        """, (result["latency_ms"], lat_breached, trace_id))
        await db.execute("""
            INSERT INTO metric_snapshots (metric, value, threshold, breached, trace_id)
            VALUES ('cost_usd', ?, 0.01, ?, ?)
        """, (result["cost_usd"], cost_breached, trace_id))
        await db.commit()

    response = AgentResponse(
        trace_id=trace_id,
        output=result["output"],
        latency_ms=result["latency_ms"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=result["cost_usd"],
    )

    await broadcast({
        "event":      "trace_completed",
        "trace_id":   trace_id,
        "latency_ms": result["latency_ms"],
        "cost_usd":   result["cost_usd"],
        "status":     "OK",
        "ts":         int(time.time()),
    })

    return response


@app.get("/api/traces")
async def list_traces(limit: int = 50, offset: int = 0):
    return get_traces(limit, offset)


@app.get("/api/traces/{trace_id}")
async def trace_detail(trace_id: str):
    t = get_trace(trace_id)
    if not t:
        raise HTTPException(404, "Trace not found")
    return t


@app.get("/api/metrics")
async def metrics():
    return get_metrics()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "vaia-l69", "ts": int(time.time())}


@app.websocket("/ws/traces")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
