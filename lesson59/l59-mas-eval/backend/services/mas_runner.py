"""
Demo MAS runner: simulates a 3-agent research pipeline with real Gemini calls.
Bridges L58 MultiModalMessage schema.
Emits TraceRecords for every inter-agent event.
"""
from __future__ import annotations
import asyncio
import os
import time
import uuid
from typing import Any

import google.generativeai as genai

from models.trace import TraceRecord, TokenCost
from core.session import DebugSession
from core.profiler import profile_async

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _estimate_tokens(text: str) -> int:
    # Approximation fallback when provider usage metadata is unavailable.
    if not text:
        return 0
    return max(1, len(text) // 4)


def _offline_fallback(agent: str, prompt: str) -> str:
    preview = " ".join(prompt.split())[:180]
    return f"{agent.title()} fallback response: {preview}"


async def _gemini_call(prompt: str, agent: str, session: DebugSession) -> str:
    """Single Gemini Flash call with full trace instrumentation."""
    span_id = str(uuid.uuid4())
    parent_id = session.records[-1].span_id if session.records else None
    start = time.monotonic_ns()

    rec = TraceRecord(
        span_id=span_id,
        parent_id=parent_id,
        agent_name=agent,
        event_type="LLM_CALL",
        payload={"prompt_preview": prompt[:200]},
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        if not prompt_tokens:
            prompt_tokens = _estimate_tokens(prompt)
        if not completion_tokens:
            completion_tokens = _estimate_tokens(text)
        rec.duration_ns = time.monotonic_ns() - start
        rec.token_cost = TokenCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        rec.payload["response_preview"] = text[:300]
        await session.emit(rec)
        return text
    except Exception as e:
        rec.event_type = "ERROR"
        rec.error_msg = str(e)
        rec.duration_ns = time.monotonic_ns() - start
        fallback = _offline_fallback(agent, prompt)
        rec.token_cost = TokenCost(
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(fallback),
        )
        rec.payload["response_preview"] = fallback[:300]
        rec.payload["fallback"] = True
        await session.emit(rec)
        return fallback


async def _send_message(from_agent: str, to_agent: str, content: str,
                        session: DebugSession) -> None:
    parent_id = session.records[-1].span_id if session.records else None
    rec = TraceRecord(
        parent_id=parent_id,
        agent_name=from_agent,
        event_type="SEND",
        payload={"to": to_agent, "content_preview": content[:200]},
    )
    await session.emit(rec)


async def run_research_pipeline(query: str, session: DebugSession) -> str:
    """
    3-agent pipeline:
      Orchestrator → Researcher → Analyst → Orchestrator (synthesis)
    """
    await session.attach(["orchestrator", "researcher", "analyst"])

    # ── Orchestrator: decompose ──────────────────────────────────────────────
    decomp = await _gemini_call(
        f"You are an orchestrator. Decompose this research query into 2 sub-tasks: '{query}'. "
        "Return a plain numbered list.",
        "orchestrator", session,
    )
    await _send_message("orchestrator", "researcher", decomp, session)

    # ── Researcher: gather information ───────────────────────────────────────
    research = await _gemini_call(
        f"You are a researcher. Address these sub-tasks concisely:\n{decomp}",
        "researcher", session,
    )
    await _send_message("researcher", "analyst", research, session)

    # ── Analyst: synthesise ──────────────────────────────────────────────────
    analysis = await _gemini_call(
        f"You are an analyst. Given this research:\n{research}\n\n"
        f"Synthesise a 3-bullet executive summary for: '{query}'",
        "analyst", session,
    )
    await _send_message("analyst", "orchestrator", analysis, session)

    # ── Orchestrator: final output ───────────────────────────────────────────
    final = await _gemini_call(
        f"You are an orchestrator. Format this analyst output as a clean final answer:\n{analysis}",
        "orchestrator", session,
    )

    session.final_output = final
    await session.close()
    return final
