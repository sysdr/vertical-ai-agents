from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from ..db.store import UserProfileStore
from ..services.inference import PreferenceInferenceEngine
from ..services.router import AdaptiveAgentRouter
from ..models.profile import PreferenceVector
from ..config import GEMINI_API_KEY

router = APIRouter(prefix="/profiles", tags=["profiles"])
logger = logging.getLogger(__name__)

store = UserProfileStore()
router_svc = AdaptiveAgentRouter()


def get_engine():
    return PreferenceInferenceEngine(store, GEMINI_API_KEY)


class CreateProfileRequest(BaseModel):
    user_id: str
    display_name: str = "Anonymous"
    email: str = ""
    explicit: dict = {}
    consent_behavioral: bool = False
    consent_embedding: bool = False


class UpdatePreferencesRequest(BaseModel):
    preferences: dict


class ConsentRequest(BaseModel):
    behavioral: bool
    embedding: bool


@router.post("")
async def create_profile(req: CreateProfileRequest):
    profile = await store.create_profile(
        req.user_id, req.display_name, req.email, req.explicit
    )
    if req.consent_behavioral or req.consent_embedding:
        await store.update_consent(req.user_id, req.consent_behavioral, req.consent_embedding)
    return profile


@router.get("")
async def list_profiles(limit: int = 50):
    return await store.list_profiles(limit)


@router.get("/{user_id}")
async def get_profile(user_id: str):
    profile = await store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found: {user_id}")
    return profile


@router.patch("/{user_id}/preferences")
async def update_preferences(user_id: str, req: UpdatePreferencesRequest):
    profile = await store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found: {user_id}")
    return await store.update_explicit_preferences(user_id, req.preferences)


@router.patch("/{user_id}/consent")
async def update_consent(user_id: str, req: ConsentRequest):
    profile = await store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found: {user_id}")
    await store.update_consent(user_id, req.behavioral, req.embedding)
    return {"status": "updated", "user_id": user_id}


@router.post("/{user_id}/infer")
async def trigger_inference(user_id: str):
    engine = get_engine()
    result = await engine.infer_and_store(user_id)
    return result


@router.get("/{user_id}/persona")
async def get_persona(user_id: str):
    profile = await store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found: {user_id}")

    pv = profile.get("preference_vector")
    if not pv:
        return {"persona": "DEFAULT", "confidence": 0.0, "scores": {}}

    vector = PreferenceVector.from_dict(pv)
    persona = router_svc.select_persona(vector)
    scores = router_svc.score_all(vector)
    best_score = max(scores.values())

    return {
        "persona": persona.value,
        "confidence": round(best_score, 4),
        "scores": scores
    }


@router.get("/{user_id}/drift")
async def get_drift_history(user_id: str, limit: int = 30):
    return await store.get_drift_history(user_id, limit)


@router.get("/{user_id}/adk-context")
async def get_adk_context(user_id: str):
    """L55 prep: export as Google ADK UserContext format."""
    from ..models.profile import UserProfile, PersonaType
    profile = await store.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found: {user_id}")

    pv_data = profile.get("preference_vector")
    pv = PreferenceVector.from_dict(pv_data) if pv_data else PreferenceVector.default()

    up = UserProfile(
        user_id=profile["user_id"],
        display_name=profile["display_name"],
        preference_vector=pv,
        persona=PersonaType(profile.get("persona", "DEFAULT")),
        interaction_count=profile.get("interaction_count", 0)
    )
    return up.to_adk_user_context()


@router.delete("/{user_id}")
async def delete_profile(user_id: str):
    """Right-to-forget implementation."""
    success = await store.delete_profile(user_id)
    return {"deleted": success, "user_id": user_id}
