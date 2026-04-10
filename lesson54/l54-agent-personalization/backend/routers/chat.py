from fastapi import APIRouter, Body, Header
from pydantic import BaseModel
import json
from typing import Optional
import logging
import google.generativeai as genai

from ..db.store import UserProfileStore
from ..services.router import AdaptiveAgentRouter
from ..services.compressor import ContextCompressor
from ..models.profile import PreferenceVector, PersonaType
from ..models.budget import BudgetState
from ..config import GEMINI_API_KEY

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

store = UserProfileStore()
router_svc = AdaptiveAgentRouter()
compressor = ContextCompressor()

genai.configure(api_key=GEMINI_API_KEY or "")
model = genai.GenerativeModel("gemini-2.0-flash")


def _offline_persona_reply(persona: str, message: str, tier: str) -> str:
    msg = message.strip()
    if persona == "EXECUTIVE":
        return (
            "[Offline demo - EXECUTIVE] Bottom line: "
            f"{msg[:120]}\n"
            "- Impact: prioritize business outcome and risk.\n"
            f"- Delivery mode: concise bullets, tier={tier}."
        )
    if persona == "PRACTITIONER":
        return (
            "[Offline demo - PRACTITIONER]\n"
            f"Task: {msg[:120]}\n"
            "1) Proposed implementation steps\n"
            "2) Edge cases + observability\n"
            f"3) Apply depth based on tier={tier}."
        )
    if persona == "LEARNER":
        return (
            "[Offline demo - LEARNER] Think of it like this: \
"
            f"{msg[:120]}\n"
            "- Start from basics\n"
            "- Add one concrete example\n"
            f"- Keep language accessible (tier={tier})."
        )
    if persona == "ANALYST":
        return (
            "[Offline demo - ANALYST] Structured analysis for: "
            f"{msg[:120]}\n"
            "- Hypothesis\n- Evidence to collect\n- Trade-offs / uncertainty\n"
            f"- Response detail tier={tier}."
        )
    return (
        "[Offline demo - DEFAULT] "
        f"{msg[:140]}\n"
        f"(Gemini unavailable; personalized routing still active, tier={tier}.)"
    )


def _generate_agent_reply(full_prompt: str, persona: str, message: str, tier: str) -> str:
    """Call Gemini when available; fallback remains persona-specific for demos."""
    try:
        response = model.generate_content(full_prompt)
        text = (response.text or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning("Gemini unavailable (%s); using offline demo reply", e)
    return _offline_persona_reply(persona, message, tier)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    budget_fraction: float = 1.0  # from L53 BudgetEnforcer


class FeedbackRequest(BaseModel):
    interaction_id: str
    score: float  # -1.0 to 1.0


class CompareRequest(BaseModel):
    message: str
    user_ids: list[str]


@router.post("")
async def chat(req: dict | str = Body(...), x_user_id: Optional[str] = Header(default=None)):
    user_id = x_user_id or "anonymous"
    if isinstance(req, str):
        req = json.loads(req)
    req = ChatRequest(**req)

    # Load profile
    profile = await store.get_profile(user_id)
    if not profile:
        # Auto-create minimal profile for anonymous users
        await store.create_profile(user_id, "Anonymous")
        profile = await store.get_profile(user_id)

    # Build budget state from L53 fraction
    budget = BudgetState(
        max_tokens=8000,
        used_tokens=int(8000 * (1 - req.budget_fraction))
    )

    # Get preference vector
    pv_data = profile.get("preference_vector")
    if pv_data:
        vector = PreferenceVector.from_dict(pv_data)
        persona = router_svc.select_persona(vector)
    else:
        vector = PreferenceVector.default()
        persona = PersonaType.DEFAULT

    # Build context (budget-aware compression)
    ctx = compressor.build_context(vector, budget, persona)

    # Build system prompt with personalization
    system_prompt = ctx.system_prefix

    full_prompt = f"{system_prompt}\n\nUser message: {req.message}"
    agent_reply = _generate_agent_reply(full_prompt, persona.value, req.message, ctx.tier)
    token_count = len(full_prompt.split()) + len(agent_reply.split())  # estimate

    # Save interaction
    interaction_id = await store.save_interaction(
        user_id=user_id,
        user_msg=req.message,
        agent_msg=agent_reply,
        persona=persona.value,
        context_tier=ctx.tier,
        token_cost=token_count,
        session_id=req.session_id
    )

    return {
        "interaction_id": interaction_id,
        "response": agent_reply,
        "persona": persona.value,
        "context_tier": ctx.tier,
        "personalization_token_cost": ctx.token_cost,
        "user_id": user_id
    }


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    await store.submit_feedback(req.interaction_id, req.score)
    return {"status": "recorded"}


@router.post("/compare")
async def compare_personas(req: CompareRequest):
    """A/B comparison: same message, different user personas."""
    results = []
    for uid in req.user_ids[:4]:  # cap at 4
        profile = await store.get_profile(uid)
        if not profile:
            continue

        pv_data = profile.get("preference_vector")
        if pv_data:
            vector = PreferenceVector.from_dict(pv_data)
            persona = router_svc.select_persona(vector)
        else:
            vector = PreferenceVector.default()
            persona = PersonaType.DEFAULT

        ctx = compressor.build_default(persona)
        full_prompt = f"{ctx.system_prefix}\n\nUser message: {req.message}"
        reply = _generate_agent_reply(full_prompt, persona.value, req.message, ctx.tier)

        results.append({
            "user_id": uid,
            "persona": persona.value,
            "context_tier": ctx.tier,
            "response": reply
        })

    return {"message": req.message, "comparisons": results}


@router.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 20):
    return await store.get_recent_interactions(user_id, limit)
