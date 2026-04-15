"""
Pytest suite for EvaluationHarness, InteractionGraph, TraceReplayEngine.
"""
import asyncio
import json
import os
import uuid
import time
import pytest
import pytest_asyncio

from models.trace import TraceRecord, TokenCost, TestCase
from core.graph import InteractionGraph
from core.harness import EvaluationHarness
from core.logger import MASTraceLogger
from core.session import DebugSession, SessionState


def make_record(agent: str, event: str, parent: str | None = None,
                dur: int = 100_000, tokens: int = 10) -> TraceRecord:
    return TraceRecord(
        trace_id="test-trace",
        agent_name=agent,
        event_type=event,
        parent_id=parent,
        duration_ns=dur,
        token_cost=TokenCost(prompt_tokens=tokens, completion_tokens=tokens // 2),
    )


# ─── InteractionGraph ────────────────────────────────────────────────────────
def test_graph_builds_dag():
    r1 = make_record("orchestrator", "LLM_CALL", dur=500_000)
    r2 = make_record("researcher", "LLM_CALL", parent=r1.span_id, dur=300_000)
    r3 = make_record("analyst", "LLM_CALL", parent=r2.span_id, dur=200_000)
    g = InteractionGraph([r1, r2, r3])
    assert len(g.G.nodes) == 3
    assert len(g.G.edges) == 2


def test_critical_path_identifies_bottleneck():
    r1 = make_record("orchestrator", "LLM_CALL", dur=100_000)
    r2 = make_record("researcher",   "LLM_CALL", parent=r1.span_id, dur=900_000)
    r3 = make_record("analyst",      "LLM_CALL", parent=r2.span_id, dur=200_000)
    g = InteractionGraph([r1, r2, r3])
    cp = g.critical_path()
    assert r2.span_id in cp, "Researcher (slowest) must be on critical path"


def test_agent_hotspots_sorted():
    r1 = make_record("agent_a", "LLM_CALL", dur=1_000_000)
    r2 = make_record("agent_b", "LLM_CALL", dur=500_000)
    r3 = make_record("agent_a", "TOOL_CALL", dur=300_000)
    g = InteractionGraph([r1, r2, r3])
    hs = g.agent_hotspots()
    keys = list(hs.keys())
    assert keys[0] == "agent_a"
    assert hs["agent_a"] == 1_300_000


def test_total_tokens():
    r1 = make_record("a", "LLM_CALL", tokens=100)
    r2 = make_record("b", "LLM_CALL", tokens=200)
    g = InteractionGraph([r1, r2])
    # each record: prompt=N, completion=N//2
    assert g.total_tokens() == (100 + 50) + (200 + 100)


# ─── EvaluationHarness ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_harness_passes_valid_session(tmp_path):
    tracer = MASTraceLogger(db_path=str(tmp_path / "t.db"))
    await tracer.start()
    session = DebugSession(tracer)
    await session.attach(["agent_a"])

    r = make_record("agent_a", "LLM_CALL")
    await session.emit(r)
    session.final_output = "The answer is 42"
    await session.close()

    harness = EvaluationHarness()
    harness.add_test(TestCase(
        name="contains_answer",
        expected_output_pattern=r"answer",
        max_tokens=50_000,
        max_latency_ms=60_000,
    ))
    report = harness.score(session)
    assert report.passed == 1
    assert report.failed == 0
    await tracer.stop()


@pytest.mark.asyncio
async def test_harness_fails_token_budget(tmp_path):
    tracer = MASTraceLogger(db_path=str(tmp_path / "t2.db"))
    await tracer.start()
    session = DebugSession(tracer)
    await session.attach(["agent_a"])

    r = make_record("agent_a", "LLM_CALL", tokens=1_000_000)  # 1M tokens
    await session.emit(r)
    session.final_output = "answer present"
    await session.close()

    harness = EvaluationHarness()
    harness.add_test(TestCase(
        name="budget_check",
        expected_output_pattern=r"answer",
        max_tokens=100,   # very tight
    ))
    report = harness.score(session)
    assert report.failed == 1
    await tracer.stop()


# ─── MASTraceLogger ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_tracer_persists_records(tmp_path):
    db = str(tmp_path / "traces.db")
    tracer = MASTraceLogger(db_path=db)
    await tracer.start()

    trace_id = str(uuid.uuid4())
    for _ in range(5):
        r = TraceRecord(trace_id=trace_id, agent_name="test", event_type="LLM_CALL")
        await tracer.emit(r)

    await asyncio.sleep(0.3)
    records = await tracer.fetch_traces(trace_id)
    assert len(records) == 5
    await tracer.stop()


# ─── Fixtures ────────────────────────────────────────────────────────────────
def test_fixture_faulty_trace(tmp_path):
    """
    EvaluationHarness must detect a trace where the output is empty
    (simulating a missing analyst step).
    """
    harness = EvaluationHarness()
    harness.add_test(TestCase(name="non_empty", expected_output_pattern=r"\w+"))

    class FakeSession:
        records = [make_record("a", "LLM_CALL")]
        final_output = ""   # intentionally empty

    report = harness.score(FakeSession())  # type: ignore
    assert report.failed == 1
