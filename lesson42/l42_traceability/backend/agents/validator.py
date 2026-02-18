"""
ValidatorAgent: risk assessment with full trace metadata.
"""
import os, re
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"

COMPLIANCE_RULES = {
    "MEDICAL": ["must_include_disclaimer", "no_diagnostic_claims"],
    "FINANCIAL": ["no_investment_advice", "must_disclose_uncertainty"],
}


@dataclass
class ValidationResult:
    verdict: str                # PASS | FAIL | CONDITIONAL
    risk_score: float           # 0.0 - 1.0
    compliance_flags: list[str]
    correction_hints: list[str]
    confidence: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "verdict": self.verdict,
                "risk_score": self.risk_score,
                "flags": self.compliance_flags,
            },
            "decision": f"VERDICT:{self.verdict} RISK:{self.risk_score:.2f}",
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
            "compliance_flags": self.compliance_flags,
        }


class ValidatorAgent:
    @trace_step("validator", capture_input_fields=["intent", "docs"])
    async def validate(
        self, docs: list[dict], intent: str, query: str
    ) -> ValidationResult:
        doc_text = " ".join(d["text"] for d in docs)
        applicable_rules = COMPLIANCE_RULES.get(intent, [])

        prompt = f"""You are a compliance validator for AI systems.
Query intent: {intent}
Retrieved context: "{doc_text[:800]}"
Applicable rules: {applicable_rules}

Respond ONLY in JSON:
{{
  "verdict": "<PASS|FAIL|CONDITIONAL>",
  "risk_score": <0.0-1.0>,
  "compliance_flags": ["<flag>"],
  "correction_hints": ["<hint>"],
  "confidence": <0.0-1.0>
}}"""
        resp = self._model().generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        import json
        data = json.loads(text)
        usage = resp.usage_metadata

        return ValidationResult(
            verdict=data.get("verdict", "PASS"),
            risk_score=float(data.get("risk_score", 0.1)),
            compliance_flags=data.get("compliance_flags", []),
            correction_hints=data.get("correction_hints", []),
            confidence=float(data.get("confidence", 0.8)),
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )

    def _model(self):
        return genai.GenerativeModel(_MODEL)
