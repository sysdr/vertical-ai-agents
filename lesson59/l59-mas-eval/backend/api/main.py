"""
FastAPI application for L59 MAS Evaluation & Debugging.
Endpoints: trace management, analysis, replay, WebSocket streaming.
"""
from __future__ import annotations
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.logger import MASTraceLogger
from core.session import DebugSession, SessionState
from core.graph import InteractionGraph
from core.harness import EvaluationHarness
from core.replay import TraceReplayEngine
from models.trace import TestCase, EvalReport
from services.mas_runner import run_research_pipeline

tracer = MASTraceLogger()
_ws_clients: set[WebSocket] = set()


async def _ws_broadcast(payload: str) -> None:
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracer.register_ws_callback(_ws_broadcast)
    await tracer.start()
    yield
    await tracer.stop()


app = FastAPI(title="L59 MAS Eval & Debug", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/traces")
async def ws_traces(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)


# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "ws_clients": len(_ws_clients), "dropped": tracer.dropped}


# ─── Run MAS pipeline ────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    query: str
    test_patterns: list[str] = []


@app.post("/api/run")
async def run_mas(req: RunRequest):
    session = DebugSession(tracer)
    harness = EvaluationHarness()
    for i, pat in enumerate(req.test_patterns):
        harness.add_test(TestCase(name=f"test_{i}", expected_output_pattern=pat))
    if not req.test_patterns:
        harness.add_test(TestCase(name="non_empty", expected_output_pattern=".+"))

    output = await run_research_pipeline(req.query, session)
    report = harness.score(session)

    graph = InteractionGraph(session.records)
    return {
        "trace_id": session.trace_id,
        "output": output,
        "eval_report": report.model_dump(),
        "graph": graph.to_dict(),
        "profiler": {},
    }


# ─── Trace queries ───────────────────────────────────────────────────────────
@app.get("/api/traces")
async def list_traces(limit: int = 50):
    return await tracer.list_traces(limit)


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    records = await tracer.fetch_traces(trace_id)
    if not records:
        raise HTTPException(404, "Trace not found")
    return records


@app.get("/api/analysis/{trace_id}")
async def analyse_trace(trace_id: str):
    raw = await tracer.fetch_traces(trace_id)
    if not raw:
        raise HTTPException(404, "Trace not found")
    graph = InteractionGraph(raw)
    return {
        "critical_path": graph.critical_path(),
        "agent_hotspots": graph.agent_hotspots(),
        "total_tokens": graph.total_tokens(),
        "graph": graph.to_dict(),
    }


@app.get("/api/analysis/{trace_id}/critical-path")
async def critical_path(trace_id: str):
    raw = await tracer.fetch_traces(trace_id)
    if not raw:
        raise HTTPException(404, "Trace not found")
    return {"critical_path": InteractionGraph(raw).critical_path()}
