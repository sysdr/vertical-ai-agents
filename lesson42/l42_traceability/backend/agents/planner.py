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
