import asyncio
import time
import google.generativeai as genai

from agent_service.config import settings
from agent_service.metrics import LLM_LATENCY, REQUESTS_TOTAL

if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

_SYSTEM = (
    "You are a production VAIA assistant deployed via an MLOps pipeline. "
    "Help users with enterprise AI architecture and MLOps questions. "
    "Be concise. Reference deployment patterns when relevant."
)


class GeminiAgent:
    def __init__(self) -> None:
        self._model = (
            genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=_SYSTEM,
            )
            if settings.gemini_api_key
            else None
        )
        self._sessions: dict[str, object] = {}

    @staticmethod
    def _fallback_response(message: str) -> str:
        msg = message.lower()
        if "rollback" in msg:
            return (
                "Rollback should trigger when key SLOs are breached for a sustained window: "
                "for example elevated p95 latency, rising error rate, or clear model drift. "
                "In this demo pipeline, automatic rollback is driven by drift/error signals that "
                "push deployment state from HEALTHY to DEGRADED, then to ROLLING_BACK."
            )
        if "drift" in msg:
            return (
                "Model drift means live inputs or outcomes differ from baseline behavior. "
                "Monitor a drift score trend, error-rate trend, and latency trend together, "
                "then alert and progressively roll back if thresholds are persistently exceeded."
            )
        if "p95" in msg or "latency" in msg or "error rate" in msg:
            return (
                "P95 latency tracks tail performance: 95% of requests complete faster than this value. "
                "Pair it with error rate for rollout decisions. In this dashboard, warn/critical behavior "
                "is driven by threshold breaches and reflected in charts, alerts, and state transitions."
            )
        if "blue-green" in msg or "deployment" in msg:
            return (
                "Blue-green deployment runs two environments (stable and candidate) and shifts "
                "traffic gradually with health checks. If metrics degrade, route traffic back to "
                "stable immediately to minimize blast radius."
            )
        return (
            "Live Gemini response is temporarily unavailable due to API quota limits. "
            "You can still use the dashboard demo flows (chat latency/error metrics, Inject Drift, "
            "state transitions, and alerts) while quota resets or billing is enabled."
        )

    async def chat(self, session_id: str, message: str, variant: str = "stable") -> dict:
        start = time.perf_counter()
        if not settings.gemini_api_key or self._model is None:
            latency = time.perf_counter() - start
            REQUESTS_TOTAL.labels(variant=variant, status="error", endpoint="chat").inc()
            return {
                "response": (
                    "GEMINI_API_KEY is not set. Configure .env for live Gemini responses; "
                    "dashboard drift/error-rate demos still work via Inject Drift."
                ),
                "latency_ms": round(latency * 1000, 2),
                "variant": variant,
                "session_id": session_id,
                "error": True,
            }
        try:
            if session_id not in self._sessions:
                self._sessions[session_id] = self._model.start_chat(history=[])
            sess = self._sessions[session_id]
            resp = await asyncio.to_thread(sess.send_message, message)
            latency = time.perf_counter() - start
            LLM_LATENCY.labels(variant=variant).observe(latency)
            REQUESTS_TOTAL.labels(variant=variant, status="success", endpoint="chat").inc()
            return {
                "response": resp.text,
                "latency_ms": round(latency * 1000, 2),
                "variant": variant,
                "session_id": session_id,
            }
        except Exception as exc:
            latency = time.perf_counter() - start
            exc_text = str(exc).lower()
            is_quota_or_rate_limit = "quota" in exc_text or "rate limit" in exc_text or "429" in exc_text
            REQUESTS_TOTAL.labels(variant=variant, status="error", endpoint="chat").inc()
            return {
                "response": (
                    self._fallback_response(message)
                    if is_quota_or_rate_limit
                    else "Agent temporarily unavailable. Please retry in a few seconds."
                ),
                "latency_ms": round(latency * 1000, 2),
                "variant": variant,
                "session_id": session_id,
                # Treat quota fallback as a served demo response so dashboard metrics keep flowing.
                "error": not is_quota_or_rate_limit,
            }
