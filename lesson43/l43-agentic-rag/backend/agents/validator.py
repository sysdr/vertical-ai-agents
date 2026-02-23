"""
L43: Validator Agent
Scores retrieved chunks for quality and risk.
Triggers self-correction loop on low quality.
"""
from __future__ import annotations
import asyncio
import time
import os
from ..core.models import (
    Chunk, ScoredChunk, ValidationResult, RetrievalResult,
    TraceEvent, PipelineState
)
from ..core.tracer import PipelineTracer
from ..core.genai_retry import with_429_retry
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

VALIDATION_THRESHOLD = 0.55
MAX_RETRIES = 3

VALIDATOR_PROMPT = """\
Score each chunk for relevance to the query. Return ONLY JSON:
{{
  "scores": [
    {{"chunk_id": "...", "quality_score": 0.0-1.0, "risk_score": 0.0-1.0}}
  ]
}}
quality_score: how relevant and useful this chunk is for answering the query.
risk_score: likelihood of hallucination or factual error if used.

Query: {query}
Chunks:
{chunks_text}"""


class ValidatorAgent:
    def __init__(self):
        self._model = genai.GenerativeModel(GEMINI_MODEL)

    async def validate(
        self,
        retrieval: RetrievalResult,
        query: str,
        tracer: PipelineTracer,
        attempt: int = 1,
    ) -> ValidationResult:
        t0 = time.time()

        if not retrieval.chunks:
            result = ValidationResult(
                validated_chunks=[], avg_quality_score=0.0,
                attempts=attempt, passed=False,
                blocked_reason="No chunks retrieved",
            )
            await self._log(tracer, query, result, t0, attempt)
            return result

        scores_map = await self._score_chunks(retrieval.chunks, query)
        scored = []
        for chunk in retrieval.chunks:
            s = scores_map.get(chunk.chunk_id, {"quality_score": chunk.score, "risk_score": 0.2})
            passed = s["quality_score"] >= VALIDATION_THRESHOLD and s["risk_score"] < 0.8
            scored.append(ScoredChunk(
                chunk=chunk,
                quality_score=s["quality_score"],
                risk_score=s["risk_score"],
                passed=passed,
            ))

        passing = [sc for sc in scored if sc.passed]
        avg_score = sum(sc.quality_score for sc in scored) / len(scored) if scored else 0.0

        result = ValidationResult(
            validated_chunks=passing or scored[:3],  # fallback: use top 3
            avg_quality_score=round(avg_score, 4),
            attempts=attempt,
            passed=avg_score >= VALIDATION_THRESHOLD,
        )
        await self._log(tracer, query, result, t0, attempt)
        return result

    async def _score_chunks(self, chunks: list[Chunk], query: str) -> dict:
        chunks_text = "\n".join([
            f"[chunk_id={c.chunk_id}] {c.text[:300]}" for c in chunks[:8]
        ])
        try:
            import json
            response = await asyncio.to_thread(
                with_429_retry,
                lambda: self._model.generate_content(
                    VALIDATOR_PROMPT.format(query=query, chunks_text=chunks_text),
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0, max_output_tokens=512,
                    ),
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return {s["chunk_id"]: s for s in data.get("scores", [])}
        except Exception:
            # Fallback: use retrieval scores
            return {c.chunk_id: {"quality_score": c.score, "risk_score": 0.2} for c in chunks}

    async def _log(self, tracer, query, result, t0, attempt):
        latency = round((time.time() - t0) * 1000, 2)
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="validator",
            state=PipelineState.VALIDATING,
            input={"query": query, "attempt": attempt},
            output={"avg_quality_score": result.avg_quality_score, "passed": result.passed,
                    "chunks_passed": len(result.validated_chunks)},
            latency_ms=latency, attempt=attempt,
        ))
