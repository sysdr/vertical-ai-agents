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
