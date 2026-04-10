"""
L54 Advanced Agent Personalization — FastAPI Application
Builds on L53 BudgetEnforcer | Prepares L55 Google ADK
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from dataclasses import asdict

from .routers import profiles_router, chat_router, analytics_router
from .db.store import UserProfileStore
from .models.profile import PreferenceVector
from .services.router import AdaptiveAgentRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="L54: Advanced Agent Personalization",
    description="VAIA Module 5 — User Profile Store, Preference Inference, Adaptive Agent Router",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profiles_router)
app.include_router(chat_router)
app.include_router(analytics_router)


@app.on_event("startup")
async def startup():
    os.makedirs("data", exist_ok=True)
    store = UserProfileStore()
    await store.init_db()

    # Seed demo profiles
    demo_profiles = [
        {"user_id": "exec-001", "display_name": "Sarah Chen (VP Eng)",
         "explicit": {"formality": 0.9, "verbosity": 0.2}, "consent_behavioral": True},
        {"user_id": "dev-002", "display_name": "Marcus Webb (SRE)",
         "explicit": {"technical_depth": 0.9, "example_density": 0.8}, "consent_behavioral": True},
        {"user_id": "junior-003", "display_name": "Priya Sharma (Jr Dev)",
         "explicit": {"technical_depth": 0.4, "example_density": 0.9}, "consent_behavioral": True},
        {"user_id": "analyst-004", "display_name": "David Kim (Data Analyst)",
         "explicit": {"hedge_tolerance": 0.8, "domain_breadth": 0.9}, "consent_behavioral": True},
    ]

    router = AdaptiveAgentRouter()

    for p in demo_profiles:
        existing = await store.get_profile(p["user_id"])
        if not existing:
            await store.create_profile(
                p["user_id"], p["display_name"],
                explicit_prefs=p.get("explicit", {})
            )
            await store.update_consent(p["user_id"], p.get("consent_behavioral", False), False)

        # Derive persona + preference vector from explicit prefs (dashboard radar / analytics)
        prof = await store.get_profile(p["user_id"])
        if prof and not prof.get("preference_vector"):
            merged = {**asdict(PreferenceVector.default()), **p.get("explicit", {})}
            pv = PreferenceVector(
                **{k: float(merged[k]) for k in PreferenceVector.__dataclass_fields__}
            )
            persona = router.select_persona(pv)
            await store.upsert_preference_vector(p["user_id"], asdict(pv), persona.value)

    # Seed sample interactions once so analytics (totals, tier bars) are non-zero on first load
    stats = await store.get_stats()
    if stats["total_interactions"] == 0:
        demo_chats = [
            ("exec-001", "What are Q4 priorities?", "Ship reliability and team velocity.", "EXECUTIVE", "RICH"),
            ("dev-002", "How does Redis persist data?", "RDB snapshots and AOF append-only files.", "PRACTITIONER", "STANDARD"),
            ("junior-003", "What is Kubernetes?", "Orchestrates containers across nodes...", "LEARNER", "RICH"),
            ("analyst-004", "Is this metric significant?", "Compare against baseline with confidence intervals.", "ANALYST", "MINIMAL"),
            ("exec-001", "Status in one line?", "Green on SLOs; one risk in dependency X.", "EXECUTIVE", "STANDARD"),
        ]
        for uid, umsg, amsg, persona_used, tier in demo_chats:
            await store.save_interaction(uid, umsg, amsg, persona_used, tier, 140, "demo-seed")

    logger.info("L54 startup complete — demo profiles seeded")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "lesson": "L54",
        "components": ["UserProfileStore", "PreferenceInferenceEngine",
                        "AdaptiveAgentRouter", "ContextCompressor"]
    }
