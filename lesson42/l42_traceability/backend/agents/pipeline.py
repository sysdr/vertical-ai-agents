"""
AgenticRAGPipeline: orchestrates Planner → Retriever → Validator → Synthesizer
with full traceability via TraceContext.
Supports DEMO_MODE for running without a valid Gemini API key.
"""
import os
import asyncio
from backend.tracing.context import TraceContext
from backend.storage.sink import AuditLogSink
from backend.agents.planner import PlannerAgent
from backend.agents.retriever import RetrieverAgent
from backend.agents.validator import ValidatorAgent
from backend.agents.synthesizer import SynthesizerAgent


def _use_demo_mode() -> bool:
    """Use mock pipeline when DEMO_MODE=1 or when GEMINI_API_KEY is missing/invalid."""
    if os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes"):
        return True
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return not key


class AgenticRAGPipeline:
    def __init__(self, audit_sink: AuditLogSink):
        self.sink = audit_sink
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.validator = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()

    async def _run_demo(self, query: str) -> dict:
        """Run a full 4-stage mock pipeline so the dashboard shows complete traces without Gemini."""
        ctx = TraceContext.mint(query=query)
        try:
            # Planner
            s1 = ctx.start_span("planner")
            await asyncio.sleep(0.06)
            s1.finish(
                status="success",
                decision="INTENT:FACTUAL STRATEGY:DEFAULT",
                confidence=0.8,
                output={"intent": "FACTUAL", "sub_queries": [query[:80] + ("..." if len(query) > 80 else "")]},
            )
            # Retriever
            s2 = ctx.start_span("retriever")
            await asyncio.sleep(0.04)
            s2.finish(
                status="success",
                decision="RETRIEVED:3 STRATEGY:DEFAULT",
                confidence=0.85,
                output={"doc_ids": ["doc_001", "doc_002", "doc_003"], "strategy": "DEFAULT", "top_score": 0.75},
            )
            # Validator
            s3 = ctx.start_span("validator")
            await asyncio.sleep(0.05)
            s3.finish(
                status="success",
                decision="VERDICT:PASS RISK:0.20",
                risk_score=0.2,
                confidence=0.9,
                output={"verdict": "PASS", "risk_score": 0.2, "flags": []},
                compliance_flags=[],
            )
            # Synthesizer
            s4 = ctx.start_span("synthesizer")
            await asyncio.sleep(0.07)
            s4.finish(
                status="success",
                decision="SYNTHESIZED citations=3 conf=0.88",
                confidence=0.88,
                output={"response_len": 200, "citation_count": 3, "confidence": 0.88},
            )
            record = ctx.finalize()
            trace_id = ctx.trace_id
            await self.sink.enqueue(record)
            ctx.reset()
            return {
                "trace_id": trace_id,
                "status": "complete",
                "intent": "FACTUAL",
                "verdict": "PASS",
                "risk_score": 0.2,
                "response": f"[Demo mode] Answer for: \"{query[:100]}\"... RAG (Retrieval-Augmented Generation) combines retrieval of relevant documents with generative AI to produce grounded, cited responses. Citations: [1] doc_001, [2] doc_002, [3] doc_003.",
                "citations": ["doc_001", "doc_002", "doc_003"],
                "reasoning_chain": ["Classified query intent as FACTUAL.", "Retrieved 3 relevant documents.", "Validated compliance (PASS, low risk).", "Synthesized response with citations."],
                "confidence_delta": record.confidence_delta,
                "span_count": 4,
                "demo_mode": True,
            }
        finally:
            try:
                ctx.reset()
            except Exception:
                pass

    async def run(self, query: str) -> dict:
        if _use_demo_mode():
            return await self._run_demo(query)

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
