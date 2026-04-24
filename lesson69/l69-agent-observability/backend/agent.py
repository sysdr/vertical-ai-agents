"""
L69 VAIA: ObservableAgent — Gemini 2.0 Flash with OTEL instrumentation.
Every execution step is a named span; LLM cost is a first-class attribute.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from opentelemetry.trace import Status, StatusCode
from opentelemetry import trace as otel_trace

from tracer import tracer

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)

# Gemini 2.0 Flash pricing (USD / 1K tokens, May 2025)
COST_INPUT_PER_1K  = 0.000075
COST_OUTPUT_PER_1K = 0.000300

_DEFAULT_SYSTEM = (
    "You are a concise, helpful VAIA assistant. "
    "Answer questions accurately in 2-4 sentences unless more detail is requested."
)


class ObservableAgent:
    """
    Gemini 2.0 Flash agent fully instrumented with OpenTelemetry.
    Produces spans: agent.run (root) → agent.build_prompt → agent.llm_call → agent.postprocess
    """

    MODEL = "gemini-2.0-flash"

    def __init__(self, system_prompt: Optional[str] = None):
        self._model = genai.GenerativeModel(
            self.MODEL,
            system_instruction=system_prompt or _DEFAULT_SYSTEM,
        )

    def run(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run agent with full observability. Returns dict with output + telemetry."""
        with tracer.start_as_current_span(
            "agent.run",
            kind=otel_trace.SpanKind.SERVER,
        ) as root:
            root.set_attribute("agent.model",       self.MODEL)
            root.set_attribute("agent.input",        user_input[:500])
            root.set_attribute("agent.context_keys", json.dumps(list(context.keys()) if context else []))

            try:
                prompt  = self._build_prompt(user_input, context)
                llm_out = self._call_llm(prompt)
                result  = self._postprocess(llm_out)

                root.set_attribute("agent.output_length",   len(result["output"]))
                root.set_attribute("agent.total_cost_usd",  result["cost_usd"])
                root.set_attribute("agent.total_latency_ms", result["latency_ms"])
                root.set_status(Status(StatusCode.OK))
                return result

            except Exception as exc:
                root.set_status(Status(StatusCode.ERROR, str(exc)))
                root.record_exception(exc)
                raise

    def _build_prompt(self, user_input: str, context: Optional[Dict]) -> str:
        with tracer.start_as_current_span("agent.build_prompt") as span:
            parts: List[str] = []
            if context:
                parts.append(f"[Context]\n{json.dumps(context, indent=2)}\n")
            parts.append(f"[Query]\n{user_input}")
            prompt = "\n".join(parts)

            span.set_attribute("prompt.length_chars", len(prompt))
            span.set_attribute("prompt.has_context",  context is not None)
            span.add_event("prompt_built", {"chars": len(prompt)})
            return prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "agent.llm_call",
            kind=otel_trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("llm.vendor",       "google")
            span.set_attribute("llm.model",        self.MODEL)
            span.set_attribute("llm.prompt_chars", len(prompt))
            span.add_event("llm_request_sent")

            t0 = time.perf_counter()
            try:
                response = self._model.generate_content(prompt)
                response_text = response.text
                usage = getattr(response, "usage_metadata", None)
                finish_reason = (
                    str(response.candidates[0].finish_reason)
                    if getattr(response, "candidates", None) else "unknown"
                )
            except Exception as exc:
                # Keep demos functional even when upstream API keys/quotas are unavailable.
                response_text = (
                    "Demo fallback response: observability tracks trace latency, "
                    "token usage, cost, and failures to make agent behavior measurable."
                )
                usage = None
                finish_reason = f"fallback:{type(exc).__name__}"
                span.add_event("llm_fallback_used", {"error": str(exc)[:200]})

            latency_ms = (time.perf_counter() - t0) * 1000
            span.add_event("llm_response_received", {"latency_ms": round(latency_ms, 2)})

            in_tok  = int(usage.prompt_token_count if usage else max(1, len(prompt.split())))
            out_tok = int(usage.candidates_token_count if usage else max(1, len(response_text.split())))
            cost    = (in_tok / 1000 * COST_INPUT_PER_1K +
                       out_tok / 1000 * COST_OUTPUT_PER_1K)

            span.set_attribute("llm.input_tokens",  in_tok)
            span.set_attribute("llm.output_tokens", out_tok)
            span.set_attribute("llm.latency_ms",    round(latency_ms, 2))
            span.set_attribute("llm.cost_usd",      round(cost, 8))
            span.set_attribute("llm.finish_reason", finish_reason)

            return {
                "text":          response_text,
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "cost_usd":      round(cost, 8),
                "latency_ms":    round(latency_ms, 2),
            }

    def _postprocess(self, llm_out: Dict) -> Dict[str, Any]:
        with tracer.start_as_current_span("agent.postprocess") as span:
            output = llm_out["text"].strip()
            span.set_attribute("output.length_chars", len(output))
            span.set_attribute("output.word_count",   len(output.split()))
            return {**llm_out, "output": output}
