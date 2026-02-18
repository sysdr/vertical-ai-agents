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
