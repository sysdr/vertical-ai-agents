"""
AuditLogSink: async dual-write to SQLite (queryable) and JSONL (immutable archive).
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
from typing import List

import aiosqlite
import aiofiles

from backend.tracing.context import TraceRecord, Span

DB_PATH = os.getenv("TRACE_DB_PATH", "data/traces/audit.db")
JSONL_PATH = os.getenv("TRACE_JSONL_PATH", "data/traces/spans.jsonl")
RING_BUFFER_SIZE = int(os.getenv("TRACE_RING_BUFFER_SIZE", "500"))


class AuditLogSink:
    """
    Dual-sink audit logger. SQLite for queryability, JSONL for immutability.
    """

    def __init__(self):
        self._ring: list[dict] = []          # in-memory ring for dashboard
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._initialized = False
        self._worker_task: asyncio.Task | None = None

    async def initialize(self):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(JSONL_PATH).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    query TEXT,
                    status TEXT,
                    confidence_delta REAL DEFAULT 0.0,
                    created_at REAL,
                    finalized_at REAL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    agent_name TEXT,
                    status TEXT,
                    decision TEXT,
                    risk_score REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    latency_ms REAL DEFAULT 0.0,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    model_id TEXT,
                    error TEXT,
                    started_at REAL,
                    ended_at REAL,
                    input_summary TEXT,
                    output_summary TEXT,
                    compliance_flags TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_spans_risk ON spans(risk_score)"
            )
            await db.commit()
        self._initialized = True
        self._worker_task = asyncio.create_task(self._flush_worker())

    async def _flush_worker(self):
        """Background worker draining the async queue."""
        while True:
            try:
                record: TraceRecord = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self._persist(record)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[AuditLogSink] flush error: {e}")

    async def enqueue(self, record: TraceRecord):
        """Non-blocking enqueue. Drops records if queue is full (metrics > audit integrity)."""
        try:
            self._queue.put_nowait(record)
            self._update_ring(record)
        except asyncio.QueueFull:
            print(f"[AuditLogSink] WARN: queue full, dropped trace {record.trace_id}")

    def _update_ring(self, record: TraceRecord):
        for span in record.spans:
            self._ring.append(span.to_dict())
            if len(self._ring) > RING_BUFFER_SIZE:
                self._ring.pop(0)

    async def _persist(self, record: TraceRecord):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, request_id, query, status, confidence_delta, created_at, finalized_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    record.trace_id, record.request_id, record.query[:1024],
                    record.status, record.confidence_delta,
                    record.created_at, record.finalized_at,
                ),
            )
            for span in record.spans:
                await db.execute(
                    """INSERT OR REPLACE INTO spans
                       (span_id, trace_id, agent_name, status, decision, risk_score, confidence,
                        latency_ms, input_tokens, output_tokens, model_id, error,
                        started_at, ended_at, input_summary, output_summary, compliance_flags)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        span.span_id, span.trace_id, span.agent_name, span.status,
                        span.decision, span.risk_score, span.confidence,
                        span.latency_ms, span.input_tokens, span.output_tokens,
                        span.model_id, span.error,
                        span.started_at, span.ended_at,
                        json.dumps(span.input_summary),
                        json.dumps(span.output_summary),
                        json.dumps(span.compliance_flags),
                    ),
                )
            await db.commit()

        # JSONL append — immutable archive
        async with aiofiles.open(JSONL_PATH, "a") as f:
            line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            await f.write(line)

    def get_ring_buffer(self) -> list[dict]:
        return list(self._ring)

    async def shutdown(self):
        if self._worker_task:
            self._worker_task.cancel()
