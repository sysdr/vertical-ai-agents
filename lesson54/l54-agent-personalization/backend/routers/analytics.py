from fastapi import APIRouter
from ..db.store import UserProfileStore

router = APIRouter(prefix="/analytics", tags=["analytics"])
store = UserProfileStore()


@router.get("/stats")
async def get_stats():
    return await store.get_stats()


@router.get("/drift/{user_id}")
async def get_drift(user_id: str):
    return await store.get_drift_history(user_id)
