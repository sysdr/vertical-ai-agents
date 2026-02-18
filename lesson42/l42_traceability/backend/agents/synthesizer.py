"""
SynthesizerAgent: CoT + citations from L41, now fully instrumented.
"""
import os, re
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"


@dataclass
class SynthesisResult:
    response: str
    citations: list[str]
    reasoning_chain: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "response_len": len(self.response),
                "citation_count": len(self.citations),
                "confidence": self.confidence,
            },
            "decision": f"SYNTHESIZED citations={len(self.citations)} conf={self.confidence:.2f}",
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
        }


class SynthesizerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel(_MODEL)

    @trace_step("synthesizer", capture_input_fields=["query"])
    async def synthesize(
        self, query: str, docs: list[dict], validation_result
    ) -> SynthesisResult:
        doc_ctx = "\n".join(
            f"[{i+1}] {d['text']}" for i, d in enumerate(docs)
        )
        correction_notes = ""
        if hasattr(validation_result, "correction_hints") and validation_result.correction_hints:
            correction_notes = "\nApply these corrections: " + "; ".join(validation_result.correction_hints)

        prompt = f"""You are a synthesis agent using Chain-of-Thought reasoning.

Query: "{query}"
Sources:
{doc_ctx}
{correction_notes}

Respond ONLY in JSON:
{{
  "reasoning_chain": ["<step 1>", "<step 2>", "<step 3>"],
  "response": "<final answer with inline citations like [1], [2]>",
  "citations": ["<doc_id or source ref>"],
  "confidence": <0.0-1.0>
}}"""
        resp = self._model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        import json
        data = json.loads(text)
        usage = resp.usage_metadata

        return SynthesisResult(
            response=data.get("response", ""),
            citations=data.get("citations", []),
            reasoning_chain=data.get("reasoning_chain", []),
            confidence=float(data.get("confidence", 0.75)),
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )
