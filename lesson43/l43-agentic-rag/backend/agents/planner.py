"""
L43: Planner Agent
Uses Gemini to decompose a user query into structured sub-queries
with retrieval strategy annotations.
"""
from __future__ import annotations
import asyncio
import json
import time
from ..core.models import PlanResult, SubQuery, TraceEvent, PipelineState
from ..core.tracer import PipelineTracer
from ..core.genai_retry import with_429_retry
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# Default to flash-lite for free-tier quota (15 RPM, 1000 RPD); override with GEMINI_MODEL
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

PLANNER_PROMPT = """\
You are an expert query planner for an enterprise RAG system.
Given the user query, decompose it into 1-3 focused sub-queries that will maximize
retrieval quality. For each sub-query, specify:
- text: the sub-query string
- strategy: one of "semantic", "keyword", "hybrid"
- priority: 1 (high), 2 (medium), or 3 (low)

Also provide:
- retrieval_mode: "sequential" or "parallel"
- reasoning: brief explanation

Respond ONLY with valid JSON matching this schema:
{{
  "original_query": "...",
  "sub_queries": [{{"text": "...", "strategy": "...", "priority": 1}}],
  "retrieval_mode": "sequential",
  "reasoning": "..."
}}

User query: {query}"""


class PlannerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel(GEMINI_MODEL)

    async def plan(self, query: str, tracer: PipelineTracer) -> PlanResult:
        t0 = time.time()
        try:
            response = await asyncio.to_thread(
                with_429_retry,
                lambda: self._model.generate_content(
                    PLANNER_PROMPT.format(query=query),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
            )
            raw = response.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            result = PlanResult(
                original_query=data.get("original_query", query),
                sub_queries=[SubQuery(**sq) for sq in data.get("sub_queries", [])],
                retrieval_mode=data.get("retrieval_mode", "sequential"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            # Fallback plan: use original query as single sub-query
            result = PlanResult(
                original_query=query,
                sub_queries=[SubQuery(text=query, strategy="semantic", priority=1)],
                retrieval_mode="sequential",
                reasoning=f"Fallback plan due to: {str(e)}",
            )

        latency = round((time.time() - t0) * 1000, 2)
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="planner",
            state=PipelineState.PLANNING,
            input={"query": query},
            output=result.model_dump(),
            latency_ms=latency,
        ))
        return result
