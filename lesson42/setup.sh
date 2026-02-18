#!/usr/bin/env bash
# =============================================================================
# L42: Traceability Layer Implementation — setup.sh
# VAIA Curriculum · Module 4 · Lesson 42
# =============================================================================
set -euo pipefail

PROJECT="l42_traceability"
# Set your GEMINI_API_KEY here or leave empty to use DEMO_MODE
GEMINI_API_KEY=""

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[L42]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# ─────────────────────────────────────────────────────────────────────────────
log "Creating project structure for L42: Traceability Layer"
mkdir -p $PROJECT/{backend/{agents,tracing,storage,api},frontend/src/{components,hooks,utils},tests,scripts,data/traces}
cd $PROJECT

# ─────────────────────────────────────────────────────────────────────────────
log "Setting up Python virtual environment"
if ! python3 -m venv venv 2>/dev/null; then
  warn "venv with ensurepip failed, trying --without-pip"
  python3 -m venv --without-pip venv
  source venv/bin/activate
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py -q
else
  source venv/bin/activate
  pip install --quiet --upgrade pip
fi
source venv/bin/activate

pip install --quiet --upgrade pip 2>/dev/null || true
pip install --quiet \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    google-generativeai==0.7.2 \
    pydantic==2.7.1 \
    structlog==24.1.0 \
    aiosqlite==0.20.0 \
    aiofiles==23.2.1 \
    httpx==0.27.0 \
    sse-starlette==2.1.0 \
    pytest==8.2.0 \
    pytest-asyncio==0.23.6 \
    python-dotenv==1.0.1 \
    rich==13.7.1 \
    uuid6==2024.1.12

log "Python dependencies installed"

# ─────────────────────────────────────────────────────────────────────────────
# .env
cat > .env <<EOF
GEMINI_API_KEY=${GEMINI_API_KEY}
TRACE_DB_PATH=data/traces/audit.db
TRACE_JSONL_PATH=data/traces/spans.jsonl
TRACE_RING_BUFFER_SIZE=500
LOG_LEVEL=INFO
PORT=8042
EOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/tracing/context.py — TraceContext + ContextVar propagation"
cat > backend/tracing/context.py <<'PYEOF'
"""
TraceContext: mint, propagate, finalize.
Uses ContextVar for implicit async-safe propagation.
"""
from __future__ import annotations
import time, uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List

# ---- UUID v7 (time-ordered) ------------------------------------------------
def _uuid7() -> str:
    """Lexicographically sortable UUID based on millisecond timestamp."""
    ms = int(time.time() * 1000)
    rand = uuid.uuid4().int & ((1 << 74) - 1)
    value = (ms << 74) | rand
    hex_str = f"{value:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-7{hex_str[13:16]}-{hex_str[16:20]}-{hex_str[20:32]}"

# ---- ContextVar ---------------------------------------------------------------
_trace_ctx: ContextVar[Optional["TraceContext"]] = ContextVar("trace_ctx", default=None)

def get_current_trace() -> "TraceContext":
    ctx = _trace_ctx.get()
    if ctx is None:
        raise RuntimeError("No TraceContext active — call TraceContext.mint() first")
    return ctx

def get_current_trace_or_none() -> Optional["TraceContext"]:
    return _trace_ctx.get()

# ---- Span dataclass ----------------------------------------------------------
@dataclass
class Span:
    trace_id: str
    span_id: str
    agent_name: str
    started_at: float
    ended_at: float = 0.0
    latency_ms: float = 0.0
    status: str = "pending"       # pending | success | error | skipped
    input_summary: dict = field(default_factory=dict)
    output_summary: dict = field(default_factory=dict)
    decision: str = ""
    risk_score: float = 0.0
    confidence: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""
    error: str = ""
    compliance_flags: List[str] = field(default_factory=list)

    def finish(
        self,
        output: dict | None = None,
        decision: str = "",
        status: str = "success",
        risk_score: float = 0.0,
        confidence: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_id: str = "",
        error: str = "",
        compliance_flags: List[str] | None = None,
    ) -> None:
        self.ended_at = time.monotonic()
        self.latency_ms = round((self.ended_at - self.started_at) * 1000, 2)
        self.status = status
        self.output_summary = output or {}
        self.decision = decision
        self.risk_score = risk_score
        self.confidence = confidence
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_id = model_id
        self.error = error
        self.compliance_flags = compliance_flags or []

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": self.model_id,
            "error": self.error,
            "compliance_flags": self.compliance_flags,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
        }

# ---- TraceRecord ------------------------------------------------------------
@dataclass
class TraceRecord:
    trace_id: str
    request_id: str
    query: str
    created_at: float
    finalized_at: float = 0.0
    status: str = "active"        # active | complete | failed
    spans: List[Span] = field(default_factory=list)
    confidence_delta: float = 0.0  # assignment extension field

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "query": self.query,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "status": self.status,
            "confidence_delta": self.confidence_delta,
            "spans": [s.to_dict() for s in self.spans],
        }

# ---- TraceContext ------------------------------------------------------------
class TraceContext:
    def __init__(self, trace_id: str, request_id: str, query: str):
        self.trace_id = trace_id
        self.request_id = request_id
        self.query = query
        self._created_at = time.monotonic()
        self._spans: List[Span] = []
        self._token = _trace_ctx.set(self)

    @classmethod
    def mint(cls, query: str) -> "TraceContext":
        trace_id = _uuid7()
        request_id = _uuid7()
        return cls(trace_id=trace_id, request_id=request_id, query=query)

    def start_span(self, agent_name: str) -> Span:
        span = Span(
            trace_id=self.trace_id,
            span_id=_uuid7(),
            agent_name=agent_name,
            started_at=time.monotonic(),
        )
        self._spans.append(span)
        return span

    def finalize(self) -> TraceRecord:
        failed = any(s.status == "error" for s in self._spans)
        record = TraceRecord(
            trace_id=self.trace_id,
            request_id=self.request_id,
            query=self.query,
            created_at=self._created_at,
            finalized_at=time.monotonic(),
            status="failed" if failed else "complete",
            spans=list(self._spans),
        )
        # Compute confidence_delta if planner + synthesizer spans exist
        planner_conf = next(
            (s.confidence for s in self._spans if s.agent_name == "planner"), None
        )
        synth_conf = next(
            (s.confidence for s in self._spans if s.agent_name == "synthesizer"), None
        )
        if planner_conf is not None and synth_conf is not None:
            record.confidence_delta = round(synth_conf - planner_conf, 4)
        return record

    def reset(self):
        _trace_ctx.reset(self._token)
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/tracing/decorator.py — @trace_step"
cat > backend/tracing/decorator.py <<'PYEOF'
"""
@trace_step: non-invasive instrumentation decorator.
Wraps any async agent method without modifying agent logic.
"""
import functools
from typing import Callable, Any
from backend.tracing.context import get_current_trace_or_none, Span


def trace_step(
    agent_name: str,
    capture_input_fields: list[str] | None = None,
    capture_output_fields: list[str] | None = None,
):
    """
    Decorator factory. Usage:

        @trace_step("planner", capture_input_fields=["query"])
        async def plan(self, query: str) -> PlanResult:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_current_trace_or_none()
            if ctx is None:
                # Tracing disabled — run normally
                return await fn(*args, **kwargs)

            span: Span = ctx.start_span(agent_name)

            # Capture input summary (first positional arg that isn't self)
            if capture_input_fields:
                for i, name in enumerate(capture_input_fields):
                    idx = i + 1  # skip self
                    if idx < len(args):
                        val = args[idx]
                        span.input_summary[name] = str(val)[:512]
                    elif name in kwargs:
                        span.input_summary[name] = str(kwargs[name])[:512]

            try:
                result = await fn(*args, **kwargs)

                # Agents return objects with .to_trace_dict() or we do best-effort
                if hasattr(result, "to_trace_dict"):
                    trace_meta = result.to_trace_dict()
                    span.finish(
                        output=trace_meta.get("output_summary", {}),
                        decision=trace_meta.get("decision", ""),
                        status="success",
                        risk_score=trace_meta.get("risk_score", 0.0),
                        confidence=trace_meta.get("confidence", 0.0),
                        input_tokens=trace_meta.get("input_tokens", 0),
                        output_tokens=trace_meta.get("output_tokens", 0),
                        model_id=trace_meta.get("model_id", ""),
                        compliance_flags=trace_meta.get("compliance_flags", []),
                    )
                else:
                    span.finish(
                        output={"result": str(result)[:256]},
                        status="success",
                    )
                return result

            except Exception as exc:
                span.finish(status="error", error=str(exc)[:512])
                raise

        return wrapper
    return decorator
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/storage/sink.py — AuditLogSink (SQLite + JSONL dual sink)"
cat > backend/storage/sink.py <<'PYEOF'
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
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/storage/query.py — TraceQueryEngine"
cat > backend/storage/query.py <<'PYEOF'
"""
TraceQueryEngine: reconstruct waterfall timelines, risk timelines, and span search.
"""
from __future__ import annotations
import json, os, time
from typing import List, Optional
import aiosqlite

DB_PATH = os.getenv("TRACE_DB_PATH", "data/traces/audit.db")


class TraceQueryEngine:
    async def get_traces(self, limit: int = 50) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def get_waterfall(self, trace_id: str) -> dict:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ) as c:
                trace = dict(await c.fetchone() or {})
            async with db.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC",
                (trace_id,),
            ) as c:
                spans = [dict(r) for r in await c.fetchall()]
        for s in spans:
            s["input_summary"] = json.loads(s.get("input_summary") or "{}")
            s["output_summary"] = json.loads(s.get("output_summary") or "{}")
            s["compliance_flags"] = json.loads(s.get("compliance_flags") or "[]")
        trace["spans"] = spans
        return trace

    async def get_risk_timeline(self, hours: int = 24) -> List[dict]:
        since = time.monotonic() - (hours * 3600)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT agent_name, AVG(risk_score) as avg_risk, MAX(risk_score) as max_risk,
                          COUNT(*) as count
                   FROM spans
                   WHERE started_at > ?
                   GROUP BY agent_name
                   ORDER BY avg_risk DESC""",
                (since,),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def search_by_decision(self, decision_fragment: str, limit: int = 20) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM spans WHERE decision LIKE ? ORDER BY started_at DESC LIMIT ?",
                (f"%{decision_fragment}%", limit),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def get_high_risk_traces(self, threshold: float = 0.7, limit: int = 20) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT DISTINCT t.* FROM traces t
                   JOIN spans s ON t.trace_id = s.trace_id
                   WHERE s.risk_score >= ?
                   ORDER BY t.created_at DESC LIMIT ?""",
                (threshold, limit),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def get_agent_latency_stats(self) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT agent_name,
                          ROUND(AVG(latency_ms),2) as avg_ms,
                          ROUND(MIN(latency_ms),2) as min_ms,
                          ROUND(MAX(latency_ms),2) as max_ms,
                          COUNT(*) as total_spans
                   FROM spans GROUP BY agent_name""",
            ) as c:
                return [dict(r) for r in await c.fetchall()]
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/agents — instrumented agents (Planner, Retriever, Validator, Synthesizer)"
cat > backend/agents/planner.py <<'PYEOF'
"""
PlannerAgent: instrumented with @trace_step.
Reuses L41 intent classification pattern + adds trace metadata.
"""
import os, time
import google.generativeai as genai
from dataclasses import dataclass
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"

INTENT_CLASSES = ["FACTUAL", "ANALYTICAL", "MEDICAL", "FINANCIAL", "GENERAL"]


@dataclass
class PlanResult:
    intent: str
    sub_queries: list[str]
    routing_strategy: str
    confidence: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {"intent": self.intent, "sub_queries": self.sub_queries[:3]},
            "decision": f"INTENT:{self.intent} STRATEGY:{self.routing_strategy}",
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
        }


class PlannerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel(_MODEL)

    @trace_step("planner", capture_input_fields=["query"])
    async def plan(self, query: str) -> PlanResult:
        prompt = f"""You are a query planner. Analyze this query and respond ONLY in JSON.
Query: "{query}"

Respond with exactly this structure:
{{
  "intent": "<one of: {', '.join(INTENT_CLASSES)}>",
  "sub_queries": ["<sub-query-1>", "<sub-query-2>"],
  "routing_strategy": "<STRICT|BROAD|DEFAULT>",
  "confidence": <0.0-1.0 float>
}}"""
        resp = self._model.generate_content(prompt)
        import json, re
        text = resp.text.strip()
        # Strip markdown fences
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)

        usage = resp.usage_metadata
        return PlanResult(
            intent=data.get("intent", "GENERAL"),
            sub_queries=data.get("sub_queries", [query]),
            routing_strategy=data.get("routing_strategy", "DEFAULT"),
            confidence=float(data.get("confidence", 0.7)),
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
PYEOF

cat > backend/agents/retriever.py <<'PYEOF'
"""
RetrieverAgent: simulates semantic retrieval with trace instrumentation.
"""
import os, hashlib, time
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"

# Simulated document corpus
CORPUS = [
    {"id": "doc_001", "text": "Neural networks learn by adjusting weights through backpropagation."},
    {"id": "doc_002", "text": "Transformer models use attention mechanisms to process sequences."},
    {"id": "doc_003", "text": "RAG systems retrieve documents before generating responses."},
    {"id": "doc_004", "text": "Enterprise AI requires compliance, auditability, and observability."},
    {"id": "doc_005", "text": "Vector databases store embeddings for semantic similarity search."},
    {"id": "doc_006", "text": "Agentic pipelines decompose complex queries into sub-tasks."},
]


@dataclass
class RetrievalResult:
    docs: list[dict]
    scores: list[float]
    strategy_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "doc_ids": [d["id"] for d in self.docs],
                "top_score": max(self.scores, default=0.0),
                "strategy": self.strategy_used,
            },
            "decision": f"RETRIEVED:{len(self.docs)} STRATEGY:{self.strategy_used}",
            "confidence": max(self.scores, default=0.0),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
        }


class RetrieverAgent:
    @trace_step("retriever", capture_input_fields=["query", "strategy"])
    async def retrieve(
        self, query: str, strategy: str = "DEFAULT", top_k: int = 3
    ) -> RetrievalResult:
        t0 = time.monotonic()

        # Simulate semantic scoring with keyword overlap
        scores = []
        query_words = set(query.lower().split())
        for doc in CORPUS:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / (len(query_words) + 1)
            scores.append(score)

        # Rerank: strict strategy boosts exact matches
        if strategy == "STRICT":
            scores = [s * 1.3 if s > 0 else 0 for s in scores]

        pairs = sorted(zip(scores, CORPUS), reverse=True)[:top_k]
        selected_scores = [p[0] for p in pairs]
        selected_docs = [p[1] for p in pairs]

        return RetrievalResult(
            docs=selected_docs,
            scores=selected_scores,
            strategy_used=strategy,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            input_tokens=len(query.split()),
            output_tokens=sum(len(d["text"].split()) for d in selected_docs),
        )
PYEOF

cat > backend/agents/validator.py <<'PYEOF'
"""
ValidatorAgent: risk assessment with full trace metadata.
"""
import os, re
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"

COMPLIANCE_RULES = {
    "MEDICAL": ["must_include_disclaimer", "no_diagnostic_claims"],
    "FINANCIAL": ["no_investment_advice", "must_disclose_uncertainty"],
}


@dataclass
class ValidationResult:
    verdict: str                # PASS | FAIL | CONDITIONAL
    risk_score: float           # 0.0 - 1.0
    compliance_flags: list[str]
    correction_hints: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "verdict": self.verdict,
                "risk_score": self.risk_score,
                "flags": self.compliance_flags,
            },
            "decision": f"VERDICT:{self.verdict} RISK:{self.risk_score:.2f}",
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
            "compliance_flags": self.compliance_flags,
        }


class ValidatorAgent:
    @trace_step("validator", capture_input_fields=["intent", "docs"])
    async def validate(
        self, docs: list[dict], intent: str, query: str
    ) -> ValidationResult:
        doc_text = " ".join(d["text"] for d in docs)
        applicable_rules = COMPLIANCE_RULES.get(intent, [])

        prompt = f"""You are a compliance validator for AI systems.
Query intent: {intent}
Retrieved context: "{doc_text[:800]}"
Applicable rules: {applicable_rules}

Respond ONLY in JSON:
{{
  "verdict": "<PASS|FAIL|CONDITIONAL>",
  "risk_score": <0.0-1.0>,
  "compliance_flags": ["<flag>"],
  "correction_hints": ["<hint>"],
  "confidence": <0.0-1.0>
}}"""
        resp = self._model().generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        import json
        data = json.loads(text)
        usage = resp.usage_metadata

        return ValidationResult(
            verdict=data.get("verdict", "PASS"),
            risk_score=float(data.get("risk_score", 0.1)),
            compliance_flags=data.get("compliance_flags", []),
            correction_hints=data.get("correction_hints", []),
            confidence=float(data.get("confidence", 0.8)),
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )

    def _model(self):
        return genai.GenerativeModel(_MODEL)
PYEOF

cat > backend/agents/synthesizer.py <<'PYEOF'
"""
SynthesizerAgent: CoT + citations from L41, now fully instrumented.
"""
import os, re
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"


@dataclass
class SynthesisResult:
    response: str
    citations: list[str]
    reasoning_chain: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "response_len": len(self.response),
                "citation_count": len(self.citations),
                "confidence": self.confidence,
            },
            "decision": f"SYNTHESIZED citations={len(self.citations)} conf={self.confidence:.2f}",
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
        }


class SynthesizerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel(_MODEL)

    @trace_step("synthesizer", capture_input_fields=["query"])
    async def synthesize(
        self, query: str, docs: list[dict], validation_result
    ) -> SynthesisResult:
        doc_ctx = "\n".join(
            f"[{i+1}] {d['text']}" for i, d in enumerate(docs)
        )
        correction_notes = ""
        if hasattr(validation_result, "correction_hints") and validation_result.correction_hints:
            correction_notes = "\nApply these corrections: " + "; ".join(validation_result.correction_hints)

        prompt = f"""You are a synthesis agent using Chain-of-Thought reasoning.

Query: "{query}"
Sources:
{doc_ctx}
{correction_notes}

Respond ONLY in JSON:
{{
  "reasoning_chain": ["<step 1>", "<step 2>", "<step 3>"],
  "response": "<final answer with inline citations like [1], [2]>",
  "citations": ["<doc_id or source ref>"],
  "confidence": <0.0-1.0>
}}"""
        resp = self._model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        import json
        data = json.loads(text)
        usage = resp.usage_metadata

        return SynthesisResult(
            response=data.get("response", ""),
            citations=data.get("citations", []),
            reasoning_chain=data.get("reasoning_chain", []),
            confidence=float(data.get("confidence", 0.75)),
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/agents/pipeline.py — orchestrated 4-agent pipeline"
cat > backend/agents/pipeline.py <<'PYEOF'
"""
AgenticRAGPipeline: orchestrates Planner → Retriever → Validator → Synthesizer
with full traceability via TraceContext.
"""
import os
from backend.tracing.context import TraceContext
from backend.storage.sink import AuditLogSink
from backend.agents.planner import PlannerAgent
from backend.agents.retriever import RetrieverAgent
from backend.agents.validator import ValidatorAgent
from backend.agents.synthesizer import SynthesizerAgent


class AgenticRAGPipeline:
    def __init__(self, audit_sink: AuditLogSink):
        self.sink = audit_sink
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.validator = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()

    async def run(self, query: str) -> dict:
        ctx = TraceContext.mint(query=query)
        try:
            # Stage 1: Plan
            plan = await self.planner.plan(query)

            # Stage 2: Retrieve
            retrieval = await self.retriever.retrieve(
                query=query, strategy=plan.routing_strategy
            )

            # Stage 3: Validate
            validation = await self.validator.validate(
                docs=retrieval.docs, intent=plan.intent, query=query
            )

            # Stage 4: Synthesize (from L41)
            synthesis = await self.synthesizer.synthesize(
                query=query, docs=retrieval.docs, validation_result=validation
            )

            record = ctx.finalize()
            await self.sink.enqueue(record)

            return {
                "trace_id": ctx.trace_id,
                "status": record.status,
                "intent": plan.intent,
                "verdict": validation.verdict,
                "risk_score": validation.risk_score,
                "response": synthesis.response,
                "citations": synthesis.citations,
                "reasoning_chain": synthesis.reasoning_chain,
                "confidence_delta": record.confidence_delta,
                "span_count": len(record.spans),
            }

        except Exception as exc:
            record = ctx.finalize()
            await self.sink.enqueue(record)
            return {
                "trace_id": ctx.trace_id,
                "status": "failed",
                "error": str(exc),
            }
        finally:
            ctx.reset()
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/api/app.py — FastAPI with SSE live tail"
cat > backend/api/app.py <<'PYEOF'
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
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing backend/__init__ files"
touch backend/__init__.py backend/tracing/__init__.py backend/storage/__init__.py backend/agents/__init__.py backend/api/__init__.py

# ─────────────────────────────────────────────────────────────────────────────
log "Writing tests/test_traceability.py"
cat > tests/test_traceability.py <<'PYEOF'
"""L42 test suite — traceability layer verification."""
import asyncio, json, os, time, pytest

os.environ.setdefault("TRACE_DB_PATH", "data/traces/test_audit.db")
os.environ.setdefault("TRACE_JSONL_PATH", "data/traces/test_spans.jsonl")
os.environ.setdefault("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

from backend.tracing.context import TraceContext, Span, get_current_trace


# ── T1: TraceContext mint + propagation ───────────────────────────────────────
def test_trace_context_mint():
    ctx = TraceContext.mint(query="test query")
    assert ctx.trace_id is not None
    assert len(ctx.trace_id) == 36
    assert get_current_trace().trace_id == ctx.trace_id
    ctx.reset()


# ── T2: Span start + finish ───────────────────────────────────────────────────
def test_span_lifecycle():
    ctx = TraceContext.mint(query="span test")
    span = ctx.start_span("planner")
    assert span.status == "pending"
    time.sleep(0.01)
    span.finish(
        output={"intent": "FACTUAL"},
        decision="INTENT:FACTUAL",
        status="success",
        confidence=0.85,
        risk_score=0.1,
    )
    assert span.status == "success"
    assert span.latency_ms >= 10
    assert span.confidence == 0.85
    ctx.reset()


# ── T3: Trace propagation across spans ───────────────────────────────────────
def test_trace_id_propagation_across_spans():
    ctx = TraceContext.mint(query="propagation test")
    span_a = ctx.start_span("planner")
    span_b = ctx.start_span("retriever")
    assert span_a.trace_id == span_b.trace_id == ctx.trace_id
    ctx.reset()


# ── T4: TraceRecord finalization ─────────────────────────────────────────────
def test_trace_finalization():
    ctx = TraceContext.mint(query="finalize test")
    s = ctx.start_span("planner")
    s.finish(status="success", confidence=0.6)
    s2 = ctx.start_span("synthesizer")
    s2.finish(status="success", confidence=0.9)
    record = ctx.finalize()
    assert record.status == "complete"
    assert len(record.spans) == 2
    assert abs(record.confidence_delta - 0.3) < 0.01
    ctx.reset()


# ── T5: Failed trace record ───────────────────────────────────────────────────
def test_failed_trace_still_persisted():
    ctx = TraceContext.mint(query="fail test")
    s = ctx.start_span("planner")
    s.finish(status="error", error="LLM timeout")
    record = ctx.finalize()
    assert record.status == "failed"
    assert record.spans[0].error == "LLM timeout"
    ctx.reset()


# ── T6: Span serialization ────────────────────────────────────────────────────
def test_span_to_dict():
    ctx = TraceContext.mint(query="serial test")
    s = ctx.start_span("validator")
    s.finish(
        output={"verdict": "PASS"},
        decision="VERDICT:PASS",
        risk_score=0.2,
        compliance_flags=["no_pii"],
    )
    d = s.to_dict()
    assert d["agent_name"] == "validator"
    assert d["risk_score"] == 0.2
    assert "no_pii" in d["compliance_flags"]
    serialized = json.dumps(d)
    assert "validator" in serialized
    ctx.reset()


# ── T7: Async sink enqueue ────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_audit_sink_enqueue():
    from backend.storage.sink import AuditLogSink
    sink = AuditLogSink()
    await sink.initialize()

    ctx = TraceContext.mint(query="sink test")
    s = ctx.start_span("planner")
    s.finish(status="success", confidence=0.7)
    record = ctx.finalize()
    await sink.enqueue(record)
    await asyncio.sleep(0.5)  # let worker flush

    ring = sink.get_ring_buffer()
    assert len(ring) >= 1
    assert ring[0]["trace_id"] == ctx.trace_id
    ctx.reset()
    await sink.shutdown()


# ── T8: JSONL consistency ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_jsonl_written():
    import os
    path = os.getenv("TRACE_JSONL_PATH", "data/traces/test_spans.jsonl")
    from backend.storage.sink import AuditLogSink
    sink = AuditLogSink()
    await sink.initialize()

    ctx = TraceContext.mint(query="jsonl test")
    s = ctx.start_span("retriever")
    s.finish(status="success")
    record = ctx.finalize()
    await sink.enqueue(record)
    await asyncio.sleep(0.8)

    assert os.path.exists(path), "JSONL file not created"
    with open(path) as f:
        lines = f.readlines()
    assert any(ctx.trace_id in line for line in lines)
    ctx.reset()
    await sink.shutdown()
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
log "Writing React frontend"
mkdir -p frontend/src/{components,hooks,utils} frontend/public

cat > frontend/package.json <<'EOF'
{
  "name": "l42-traceability-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version"]
  },
  "proxy": "http://localhost:8042"
}
EOF

cat > frontend/public/index.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>L42 Traceability Dashboard</title>
    <style>
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; background: #0f1117; }
    </style>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
EOF

cat > frontend/src/index.js <<'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
EOF

cat > frontend/src/App.js <<'JSEOF'
import React, { useState, useEffect, useRef, useCallback } from 'react';

const API = 'http://localhost:8042';

// ── Palette ───────────────────────────────────────────────────────────────────
const colors = {
  bg: '#0f1117',
  card: '#1a1d2e',
  border: '#2a2d3e',
  primary: '#4F9CF9',
  success: '#4ade80',
  warn: '#f59e0b',
  danger: '#f87171',
  muted: '#6b7280',
  text: '#e2e8f0',
  subtext: '#94a3b8',
};

const AGENT_COLORS = {
  planner: '#818cf8',
  retriever: '#34d399',
  validator: '#fbbf24',
  synthesizer: '#60a5fa',
};

// ── Utilities ─────────────────────────────────────────────────────────────────
function riskColor(score) {
  if (score >= 0.7) return colors.danger;
  if (score >= 0.4) return colors.warn;
  return colors.success;
}

function statusBadge(status) {
  const map = { complete: colors.success, failed: colors.danger, active: colors.primary };
  return map[status] || colors.muted;
}

// ── Components ────────────────────────────────────────────────────────────────
function Card({ title, children, style }) {
  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.border}`,
      borderRadius: 12, padding: '20px', marginBottom: 16, ...style
    }}>
      {title && (
        <div style={{ fontSize: 13, fontWeight: 700, color: colors.subtext, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}44`,
      borderRadius: 999, padding: '2px 10px', fontSize: 11, fontWeight: 700,
    }}>{text}</span>
  );
}

// ── Query Panel ───────────────────────────────────────────────────────────────
function QueryPanel({ onResult }) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const EXAMPLES = [
    'Explain how transformer attention works',
    'What is RAG and why is it used?',
    'How do vector databases enable semantic search?',
    'What compliance frameworks apply to enterprise AI?',
  ];

  const submit = async () => {
    if (!query.trim()) return;
    setLoading(true); setError('');
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      onResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title="🔍 Query the Agentic RAG Pipeline">
      <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && submit()}
          placeholder="Enter query..."
          style={{
            flex: 1, background: '#252836', border: `1px solid ${colors.border}`,
            borderRadius: 8, padding: '10px 14px', color: colors.text, fontSize: 14,
            outline: 'none',
          }}
        />
        <button
          onClick={submit}
          disabled={loading}
          style={{
            background: loading ? colors.muted : colors.primary,
            color: '#fff', border: 'none', borderRadius: 8,
            padding: '10px 22px', fontSize: 14, fontWeight: 700, cursor: 'pointer',
          }}
        >
          {loading ? '…' : 'Run'}
        </button>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {EXAMPLES.map(ex => (
          <button key={ex} onClick={() => setQuery(ex)} style={{
            background: '#252836', border: `1px solid ${colors.border}`,
            borderRadius: 6, padding: '4px 10px', fontSize: 11, color: colors.subtext,
            cursor: 'pointer',
          }}>{ex}</button>
        ))}
      </div>
      {error && <div style={{ color: colors.danger, fontSize: 12, marginTop: 8 }}>{error}</div>}
    </Card>
  );
}

// ── Result Panel ──────────────────────────────────────────────────────────────
function ResultPanel({ result }) {
  if (!result) return null;
  return (
    <Card title="✅ Pipeline Result">
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <Badge text={result.status?.toUpperCase()} color={statusBadge(result.status)} />
        {result.intent && <Badge text={`Intent: ${result.intent}`} color={colors.primary} />}
        {result.verdict && <Badge text={`Verdict: ${result.verdict}`} color={riskColor(result.risk_score || 0)} />}
        {result.risk_score != null && (
          <Badge text={`Risk: ${(result.risk_score * 100).toFixed(0)}%`} color={riskColor(result.risk_score)} />
        )}
        {result.confidence_delta != null && (
          <Badge text={`Δconf: ${result.confidence_delta >= 0 ? '+' : ''}${result.confidence_delta.toFixed(3)}`} color={colors.primary} />
        )}
      </div>

      {result.trace_id && (
        <div style={{ fontSize: 11, color: colors.muted, marginBottom: 10 }}>
          trace_id: <span style={{ color: colors.primary, fontFamily: 'monospace' }}>{result.trace_id}</span>
        </div>
      )}

      {result.reasoning_chain && result.reasoning_chain.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, color: colors.subtext, marginBottom: 6, fontWeight: 700 }}>Chain-of-Thought</div>
          {result.reasoning_chain.map((step, i) => (
            <div key={i} style={{
              display: 'flex', gap: 10, marginBottom: 4, fontSize: 12, color: colors.text,
              background: '#252836', borderRadius: 6, padding: '6px 10px',
            }}>
              <span style={{ color: colors.primary, fontWeight: 700 }}>Step {i + 1}</span>
              {step}
            </div>
          ))}
        </div>
      )}

      {result.response && (
        <div style={{
          background: '#252836', borderRadius: 8, padding: '12px 14px',
          fontSize: 14, color: colors.text, lineHeight: 1.6,
        }}>{result.response}</div>
      )}

      {result.citations && result.citations.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, color: colors.subtext }}>
          Citations: {result.citations.map((c, i) => (
            <span key={i} style={{ color: colors.primary, marginRight: 8 }}>[{i + 1}] {c}</span>
          ))}
        </div>
      )}
    </Card>
  );
}

// ── Waterfall Panel ───────────────────────────────────────────────────────────
function WaterfallPanel({ traceId }) {
  const [trace, setTrace] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!traceId) return;
    setLoading(true);
    fetch(`${API}/trace/${traceId}`)
      .then(r => r.json())
      .then(d => { setTrace(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [traceId]);

  if (!traceId) return null;
  if (loading) return <Card title="⏱ Trace Waterfall"><div style={{ color: colors.muted }}>Loading…</div></Card>;
  if (!trace?.spans?.length) return <Card title="⏱ Trace Waterfall"><div style={{ color: colors.muted }}>No spans found.</div></Card>;

  const totalMs = trace.spans.reduce((s, sp) => s + (sp.latency_ms || 0), 0);

  return (
    <Card title="⏱ Trace Waterfall">
      <div style={{ marginBottom: 8, fontSize: 11, color: colors.muted }}>
        Total pipeline: <strong style={{ color: colors.text }}>{totalMs.toFixed(1)}ms</strong>
        &nbsp;·&nbsp;
        {trace.spans.length} spans
        &nbsp;·&nbsp;
        Status: <strong style={{ color: statusBadge(trace.status) }}>{trace.status}</strong>
      </div>
      {trace.spans.map((span, i) => {
        const pct = totalMs > 0 ? (span.latency_ms / totalMs) * 100 : 10;
        const agentColor = AGENT_COLORS[span.agent_name] || colors.primary;
        return (
          <div key={span.span_id} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 12 }}>
              <span style={{ color: agentColor, fontWeight: 700, textTransform: 'capitalize' }}>
                {span.agent_name}
              </span>
              <span style={{ color: colors.subtext }}>
                {span.latency_ms}ms
                {span.decision && ` · ${span.decision.substring(0, 50)}`}
              </span>
            </div>
            <div style={{ background: '#252836', borderRadius: 4, height: 10, overflow: 'hidden' }}>
              <div style={{
                width: `${Math.max(pct, 2)}%`, height: '100%',
                background: agentColor, borderRadius: 4,
                opacity: span.status === 'error' ? 0.4 : 1,
              }} />
            </div>
            {span.risk_score > 0 && (
              <div style={{ fontSize: 10, color: riskColor(span.risk_score), marginTop: 2 }}>
                Risk: {(span.risk_score * 100).toFixed(0)}%
                {span.compliance_flags?.length > 0 && ` · flags: ${JSON.parse(span.compliance_flags || '[]').join(', ')}`}
              </div>
            )}
          </div>
        );
      })}
    </Card>
  );
}

// ── Risk Heatmap Panel ────────────────────────────────────────────────────────
function RiskHeatmap() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const load = () => fetch(`${API}/stats/risk-timeline`).then(r => r.json()).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const allZero = data.length > 0 && data.every(d => (d.max_risk || 0) === 0 && (d.avg_risk || 0) === 0);
  return (
    <Card title="🌡 Risk Heatmap (24h)">
      {data.length === 0
        ? <div style={{ color: colors.muted, fontSize: 12 }}>Run some queries to populate risk data.</div>
        : allZero
        ? <div style={{ color: colors.muted, fontSize: 12 }}>Run a query to see risk scores (validator/synthesizer report risk).</div>
        : data.map(d => (
          <div key={d.agent_name} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
              <span style={{ color: AGENT_COLORS[d.agent_name] || colors.primary, fontWeight: 700, textTransform: 'capitalize' }}>
                {d.agent_name}
              </span>
              <span style={{ color: riskColor(d.avg_risk) }}>
                avg {(d.avg_risk * 100).toFixed(1)}% · max {(d.max_risk * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ background: '#252836', borderRadius: 4, height: 10 }}>
              <div style={{
                width: `${Math.min(d.max_risk * 100, 100)}%`, height: '100%',
                background: riskColor(d.max_risk), borderRadius: 4, opacity: 0.8,
              }} />
            </div>
          </div>
        ))
      }
    </Card>
  );
}

// ── Latency Stats ──────────────────────────────────────────────────────────────
function LatencyStats() {
  const [data, setData] = useState([]);
  useEffect(() => {
    const load = () => fetch(`${API}/stats/latency`).then(r => r.json()).then(setData).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <Card title="⚡ Agent Latency Stats">
      {data.length === 0
        ? <div style={{ color: colors.muted, fontSize: 12 }}>No latency data yet.</div>
        : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: colors.subtext }}>
                {['Agent', 'Avg ms', 'Min ms', 'Max ms', 'Spans'].map(h => (
                  <th key={h} style={{ textAlign: 'left', paddingBottom: 8, fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map(d => (
                <tr key={d.agent_name} style={{ borderTop: `1px solid ${colors.border}` }}>
                  <td style={{ color: AGENT_COLORS[d.agent_name] || colors.primary, padding: '6px 0', textTransform: 'capitalize', fontWeight: 700 }}>
                    {d.agent_name}
                  </td>
                  <td style={{ color: colors.text }}>{d.avg_ms}</td>
                  <td style={{ color: colors.success }}>{d.min_ms}</td>
                  <td style={{ color: colors.warn }}>{d.max_ms}</td>
                  <td style={{ color: colors.subtext }}>{d.total_spans}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    </Card>
  );
}

// ── Live Tail ─────────────────────────────────────────────────────────────────
function LiveTail() {
  const [spans, setSpans] = useState([]);
  const ref = useRef(null);

  useEffect(() => {
    const es = new EventSource(`${API}/live-tail`);
    es.onmessage = e => {
      try {
        const span = JSON.parse(e.data);
        setSpans(prev => [span, ...prev].slice(0, 40));
      } catch {}
    };
    return () => es.close();
  }, []);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = 0;
  }, [spans]);

  return (
    <Card title="📡 Live Tail" style={{ maxHeight: 320, overflow: 'hidden' }}>
      <div ref={ref} style={{ maxHeight: 260, overflowY: 'auto' }}>
        {spans.length === 0
          ? <div style={{ color: colors.muted, fontSize: 12 }}>Waiting for spans…</div>
          : spans.map((s, i) => (
            <div key={s.span_id + i} style={{
              display: 'flex', gap: 10, padding: '5px 0',
              borderBottom: `1px solid ${colors.border}`, fontSize: 11, color: colors.text,
            }}>
              <span style={{ color: AGENT_COLORS[s.agent_name] || colors.primary, fontWeight: 700, minWidth: 80, textTransform: 'capitalize' }}>
                {s.agent_name}
              </span>
              <span style={{ color: statusBadge(s.status), minWidth: 50 }}>{s.status}</span>
              <span style={{ color: colors.subtext, minWidth: 60 }}>{s.latency_ms}ms</span>
              <span style={{ color: colors.muted, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.decision || s.trace_id?.slice(0, 20)}
              </span>
            </div>
          ))
        }
      </div>
    </Card>
  );
}

// ── Trace History ─────────────────────────────────────────────────────────────
function TraceHistory({ onSelect }) {
  const [traces, setTraces] = useState([]);

  const refresh = useCallback(() => {
    fetch(`${API}/traces?limit=20`).then(r => r.json()).then(setTraces).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  return (
    <Card title="📋 Trace History">
      <div style={{ maxHeight: 200, overflowY: 'auto' }}>
        {traces.length === 0
          ? <div style={{ color: colors.muted, fontSize: 12 }}>No traces yet.</div>
          : traces.map(t => (
            <div
              key={t.trace_id}
              onClick={() => onSelect(t.trace_id)}
              style={{
                display: 'flex', justifyContent: 'space-between', padding: '8px 0',
                borderBottom: `1px solid ${colors.border}`, cursor: 'pointer',
                fontSize: 12,
              }}
            >
              <span style={{ color: colors.primary, fontFamily: 'monospace', fontSize: 11 }}>
                {t.trace_id?.slice(0, 22)}…
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <Badge text={t.status} color={statusBadge(t.status)} />
                {t.confidence_delta != null && t.confidence_delta !== 0 && (
                  <Badge
                    text={`Δ${t.confidence_delta >= 0 ? '+' : ''}${t.confidence_delta.toFixed(2)}`}
                    color={t.confidence_delta >= 0 ? colors.success : colors.warn}
                  />
                )}
              </div>
            </div>
          ))
        }
      </div>
    </Card>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [result, setResult] = useState(null);
  const [selectedTrace, setSelectedTrace] = useState(null);

  const handleResult = useCallback(r => {
    setResult(r);
    setSelectedTrace(r.trace_id);
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: colors.bg, color: colors.text, padding: '24px' }}>
      {/* Header */}
      <div style={{
        textAlign: 'center', marginBottom: 28, padding: '18px',
        background: colors.card, borderRadius: 14,
        border: `1px solid ${colors.border}`,
      }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: colors.primary }}>
          L42 · Traceability Layer
        </div>
        <div style={{ fontSize: 13, color: colors.subtext, marginTop: 4 }}>
          Full audit trail: Planner intent → Validator risk → Synthesizer response
        </div>
      </div>

      {/* Query + Result */}
      <QueryPanel onResult={handleResult} />
      <ResultPanel result={result} />

      {/* Waterfall + History */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <WaterfallPanel traceId={selectedTrace} />
        <TraceHistory onSelect={setSelectedTrace} />
      </div>

      {/* Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <RiskHeatmap />
        <LatencyStats />
        <LiveTail />
      </div>
    </div>
  );
}
JSEOF

# Install frontend deps so start.sh can run npm start
if command -v npm >/dev/null 2>&1; then
  (cd frontend && npm install --silent 2>/dev/null) && log "Frontend npm dependencies installed" || warn "npm install in frontend failed (run: cd frontend && npm install)"
else
  warn "npm not found; run 'cd frontend && npm install' before scripts/start.sh"
fi

# ─────────────────────────────────────────────────────────────────────────────
log "Writing helper scripts"

cat > scripts/build.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[build] Installing frontend dependencies..."
cd frontend && npm install --silent && npm run build
echo "[build] Done. Frontend built to frontend/build/"
BASH

cat > scripts/start.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
mkdir -p data/traces

# Start backend
echo "[start] Starting FastAPI backend on :8042"
uvicorn backend.api.app:app --host 0.0.0.0 --port 8042 --reload &
BACKEND_PID=$!
echo $BACKEND_PID > .backend.pid

# Start frontend (dev)
echo "[start] Starting React frontend on :3042"
cd frontend
PORT=3042 npm start &
FE_PID=$!
echo $FE_PID > ../.frontend.pid

echo "[start] Backend PID: $BACKEND_PID | Frontend PID: $FE_PID"
echo "[start] Dashboard: http://localhost:3042"
echo "[start] API docs:   http://localhost:8042/docs"
BASH

cat > scripts/stop.sh <<'BASH'
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[ -f .backend.pid ] && kill $(cat .backend.pid) 2>/dev/null; rm -f .backend.pid
[ -f .frontend.pid ] && kill $(cat .frontend.pid) 2>/dev/null; rm -f .frontend.pid
echo "[stop] Processes stopped."
BASH

cat > scripts/test.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
mkdir -p data/traces
echo "[test] Running L42 test suite..."
python -m pytest tests/ -v --tb=short --asyncio-mode=auto
echo "[test] All tests passed ✓"
BASH

chmod +x scripts/*.sh

# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/traces

ENV TRACE_DB_PATH=data/traces/audit.db
ENV TRACE_JSONL_PATH=data/traces/spans.jsonl
ENV PORT=8042

EXPOSE 8042
CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8042"]
EOF

cat > requirements.txt <<'EOF'
fastapi==0.111.0
uvicorn[standard]==0.29.0
google-generativeai==0.7.2
pydantic==2.7.1
structlog==24.1.0
aiosqlite==0.20.0
aiofiles==23.2.1
httpx==0.27.0
sse-starlette==2.1.0
pytest==8.2.0
pytest-asyncio==0.23.6
python-dotenv==1.0.1
rich==13.7.1
uuid6==2024.1.12
EOF

cat > docker-compose.yml <<'EOF'
version: "3.9"
services:
  backend:
    build: .
    ports:
      - "8042:8042"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - TRACE_DB_PATH=/data/traces/audit.db
      - TRACE_JSONL_PATH=/data/traces/spans.jsonl
    volumes:
      - ./data:/data
    restart: unless-stopped
EOF

cat > pytest.ini <<'EOF'
[pytest]
asyncio_mode = auto
testpaths = tests
EOF

# ─────────────────────────────────────────────────────────────────────────────
log "Running quick smoke test (no LLM calls)..."
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
mkdir -p data/traces

python -c "
import asyncio, sys
sys.path.insert(0, '.')

from backend.tracing.context import TraceContext, get_current_trace
import time

# Test mint
ctx = TraceContext.mint(query='smoke test')
assert ctx.trace_id is not None, 'trace_id missing'
assert get_current_trace().trace_id == ctx.trace_id, 'ContextVar not set'

# Test span lifecycle
span = ctx.start_span('planner')
time.sleep(0.005)
span.finish(status='success', confidence=0.8, risk_score=0.1, decision='INTENT:FACTUAL')
assert span.status == 'success'
assert span.latency_ms >= 4

# Test finalization
span2 = ctx.start_span('synthesizer')
span2.finish(status='success', confidence=0.95)
record = ctx.finalize()
assert record.status == 'complete'
assert len(record.spans) == 2
assert abs(record.confidence_delta - 0.15) < 0.01
ctx.reset()

print('✓ TraceContext mint, propagation, span lifecycle, finalization OK')
print('✓ confidence_delta computation OK')
print('✓ Smoke test passed')
"

log "Smoke test passed ✓"

# ─────────────────────────────────────────────────────────────────────────────
log "Copying SVG diagrams into project"
mkdir -p docs
# Diagrams expected alongside setup.sh, copy if present
for f in architecture.svg workflow.svg state-diagram.svg; do
  [ -f "../$f" ] && cp "../$f" "docs/$f" && info "Copied $f" || warn "$f not found in parent dir (add manually to docs/)"
done

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  L42 Traceability Layer — Setup Complete  ✓${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Project dir:  $(pwd)"
echo -e "  Backend API:  http://localhost:8042"
echo -e "  API docs:     http://localhost:8042/docs"
echo -e "  Dashboard:    http://localhost:3042"
echo ""
echo -e "  ${YELLOW}Quick start:${NC}"
echo -e "    cd $PROJECT"
echo -e "    bash scripts/start.sh       # start backend + frontend"
echo -e "    bash scripts/test.sh        # run full test suite"
echo -e "    bash scripts/build.sh       # build production frontend"
echo -e "    docker-compose up --build   # Docker path"
echo ""
echo -e "  ${YELLOW}Key endpoints:${NC}"
echo -e "    POST /query               run the 4-agent traced pipeline"
echo -e "    GET  /trace/{id}          waterfall for a trace"
echo -e "    GET  /traces              list recent traces"
echo -e "    GET  /stats/risk-timeline agent risk heatmap"
echo -e "    GET  /stats/latency       per-agent latency stats"
echo -e "    GET  /live-tail           SSE stream of spans"
echo ""
echo -e "  ${BLUE}Next: L43 — Agentic RAG End-to-End${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
deactivate
cd ..