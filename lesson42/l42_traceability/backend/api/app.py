"""
FastAPI application: /query, /trace/{id}, /traces, /live-tail (SSE), /stats
"""
import asyncio, os, json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.storage.sink import AuditLogSink
from backend.storage.query import TraceQueryEngine
from backend.agents.pipeline import AgenticRAGPipeline

app = FastAPI(title="L42 Traceability Layer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

sink = AuditLogSink()
query_engine = TraceQueryEngine()
pipeline: AgenticRAGPipeline | None = None


@app.on_event("startup")
async def startup():
    global pipeline
    await sink.initialize()
    pipeline = AgenticRAGPipeline(audit_sink=sink)


@app.on_event("shutdown")
async def shutdown():
    await sink.shutdown()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def run_query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    return await pipeline.run(req.query)


@app.get("/traces")
async def list_traces(limit: int = 50):
    return await query_engine.get_traces(limit=limit)


@app.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    data = await query_engine.get_waterfall(trace_id)
    if not data:
        raise HTTPException(404, "Trace not found")
    return data


@app.get("/traces/high-risk")
async def high_risk(threshold: float = 0.7, limit: int = 20):
    return await query_engine.get_high_risk_traces(threshold=threshold, limit=limit)


@app.get("/stats/risk-timeline")
async def risk_timeline(hours: int = 24):
    return await query_engine.get_risk_timeline(hours=hours)


@app.get("/stats/latency")
async def latency_stats():
    return await query_engine.get_agent_latency_stats()


@app.get("/search/decision")
async def search_decision(q: str, limit: int = 20):
    return await query_engine.search_by_decision(q, limit=limit)


@app.get("/live-tail")
async def live_tail():
    """Server-Sent Events stream of recent spans for dashboard live tail."""
    async def generate():
        seen = set()
        while True:
            ring = sink.get_ring_buffer()
            for span in ring:
                sid = span.get("span_id", "")
                if sid not in seen:
                    seen.add(sid)
                    yield {"data": json.dumps(span)}
            await asyncio.sleep(0.3)
    return EventSourceResponse(generate())


@app.get("/health")
async def health():
    return {"status": "ok", "lesson": "L42"}
