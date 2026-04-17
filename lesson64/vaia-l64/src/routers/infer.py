"""POST /v1/agent/infer endpoint."""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("vaia.infer")
router = APIRouter()


def _looks_like_provider_auth_error(message: str) -> bool:
    upper_msg = message.upper()
    markers = ("API_KEY_INVALID", "API KEY EXPIRED", "API KEY INVALID", "UNAUTHENTICATED")
    return any(marker in upper_msg for marker in markers)


def _looks_like_provider_quota_error(message: str) -> bool:
    upper_msg = message.upper()
    markers = ("QUOTA EXCEEDED", "RESOURCE_EXHAUSTED", "RATE LIMIT", "429")
    return any(marker in upper_msg for marker in markers)


class InferRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    context: str | None = Field(None, max_length=8192)
    session_id: str | None = None


class InferResponse(BaseModel):
    answer: str
    model: str
    finish_reason: str
    session_id: str | None = None


@router.post("/infer", response_model=InferResponse)
async def agent_infer(payload: InferRequest):
    """
    Single-turn agent inference.
    L65 will load-test this endpoint with locust.
    """
    from src.main import agent  # import here to avoid circular dep at module load
    if agent is None or not agent.is_ready:
        raise HTTPException(status_code=503, detail="Agent warming up")
    try:
        result = await agent.infer(payload.query, payload.context)
        return InferResponse(
            answer=result["answer"],
            model=result["model"],
            finish_reason=result["finish_reason"],
            session_id=payload.session_id,
        )
    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"Inference failed: {err_msg}")
        if _looks_like_provider_auth_error(err_msg):
            raise HTTPException(
                status_code=503,
                detail="Model provider credentials are invalid or expired. Update GEMINI_API_KEY and retry.",
            )
        if _looks_like_provider_quota_error(err_msg):
            logger.warning("Falling back to demo response due to provider quota limits.")
            return InferResponse(
                answer=(
                    "Demo mode response: Gemini quota is currently exhausted for this key/project. "
                    "Please enable quota or billing to restore live model answers."
                ),
                model="demo-fallback",
                finish_reason="QUOTA_EXCEEDED_DEMO_FALLBACK",
                session_id=payload.session_id,
            )
        raise HTTPException(status_code=500, detail="Inference failed due to an internal server error.")
