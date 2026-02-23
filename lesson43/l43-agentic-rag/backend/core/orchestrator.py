"""
L43: Pipeline Orchestrator
Coordinates all four agents, manages state transitions, and handles
the self-correction loop with bounded retries.
Fast mode (default): retrieve -> synthesize only (1 Gemini call). Full mode: plan -> retrieve -> validate -> synthesize.
"""
from __future__ import annotations
import os
import time
from ..core.models import (
    PipelineState, SynthesisResult, QueryRequest,
    PlanResult, SubQuery, ScoredChunk,
)
from ..core.tracer import PipelineTracer
from ..agents.planner    import PlannerAgent
from ..agents.retriever  import RetrieverAgent
from ..agents.validator  import ValidatorAgent
from ..agents.synthesizer import SynthesizerAgent


def _fast_mode() -> bool:
    return os.environ.get("FAST_MODE", "1").strip().lower() in ("1", "true", "yes")


class AgenticRAGOrchestrator:
    def __init__(self, collection_name: str = "enterprise_kb"):
        self.planner    = PlannerAgent()
        self.retriever  = RetrieverAgent(collection_name)
        self.validator  = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()

    async def run(self, request: QueryRequest) -> SynthesisResult:
        tracer = PipelineTracer()
        state  = PipelineState.IDLE
        t0     = time.time()
        top_k  = min(request.max_results, 5)

        try:
            if _fast_mode():
                # Fast path: query -> retrieve -> synthesize (1 Gemini call, ~5–15s)
                plan = PlanResult(
                    original_query=request.query,
                    sub_queries=[SubQuery(text=request.query, strategy="semantic", priority=1)],
                    retrieval_mode="sequential",
                    reasoning="fast",
                )
                state = PipelineState.RETRIEVING
                retrieval = await self.retriever.retrieve(plan, tracer, attempt=1, top_k=top_k)
                # Treat top chunks as validated (no validator call)
                validated = [
                    ScoredChunk(chunk=c, quality_score=c.score, risk_score=0.2, passed=True)
                    for c in (retrieval.chunks[:top_k] or retrieval.chunks[:3])
                ]
                state = PipelineState.SYNTHESIZING
                result = await self.synthesizer.synthesize(request.query, validated, tracer)
            else:
                # Full path: plan -> retrieve -> validate (loop) -> synthesize
                state = PipelineState.PLANNING
                plan = await self.planner.plan(request.query, tracer)
                state = PipelineState.RETRIEVING
                retrieval = await self.retriever.retrieve(plan, tracer, attempt=1, top_k=top_k)
                state = PipelineState.VALIDATING
                validation = await self.validator.validate(retrieval, request.query, tracer, attempt=1)
                max_retries = 3
                attempt = 1
                while not validation.passed and attempt < max_retries:
                    attempt += 1
                    state = PipelineState.RETRIEVING
                    retrieval = await self.retriever.retrieve(plan, tracer, attempt=attempt, top_k=top_k)
                    state = PipelineState.VALIDATING
                    validation = await self.validator.validate(retrieval, request.query, tracer, attempt=attempt)
                state = PipelineState.SYNTHESIZING
                result = await self.synthesizer.synthesize(
                    request.query,
                    validation.validated_chunks,
                    tracer,
                )

            # ─ TRACING ─
            state = PipelineState.TRACING
            try:
                await tracer.finalize(PipelineState.DONE, result.model_dump(mode="json"))
            except Exception:
                pass  # ensure we always return the result
            result.pipeline_state = PipelineState.DONE
            return result

        except Exception as e:
            try:
                await tracer.finalize(PipelineState.ERROR, {"error": str(e)})
            except Exception:
                pass
            return SynthesisResult(
                trace_id=tracer.trace_id,
                query=request.query,
                response=f"Pipeline error in {state.value}: {str(e)}",
                citations=[],
                tokens_used=0,
                latency_ms=round((time.time() - t0) * 1000, 2),
                quality_score=0.0,
                pipeline_state=PipelineState.ERROR,
            )
