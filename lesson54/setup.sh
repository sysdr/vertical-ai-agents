#!/usr/bin/env bash
# =============================================================================
# L54: Advanced Agent Personalization — Full Project Setup
# VAIA Curriculum: 90-Lesson Code-First AI Engineering Series
# Builds on: L53 BudgetEnforcer | Prepares: L55 Google ADK
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/l54-agent-personalization"

log()  { echo -e "${GREEN}[L54]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

banner() {
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║         L54: Advanced Agent Personalization                    ║
║         VAIA — Multi-Agent Architectures Module 5              ║
║         Builds on L53 BudgetEnforcer → Prepares L55 ADK        ║
╚═══════════════════════════════════════════════════════════════╝
EOF
}

banner

# =============================================================================
# STEP 1: Project Structure
# =============================================================================
step "Creating project structure"

mkdir -p "$PROJECT_ROOT"/{backend,frontend/src/{components,hooks,utils},data,scripts,docker,diagrams}
mkdir -p "$PROJECT_ROOT"/backend/{routers,services,models,middleware,db,tasks}
cd "$PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"

# =============================================================================
# STEP 2: Backend — Models
# =============================================================================
step "Writing backend models"

cat > backend/models/__init__.py << 'PYEOF'
from .profile import UserProfile, PreferenceVector, PersonaType, PersonalizationContext
from .interaction import Interaction, InteractionFeedback
from .budget import BudgetState
PYEOF

cat > backend/models/profile.py << 'PYEOF'
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import numpy as np
import json
import base64


class PersonaType(str, Enum):
    EXECUTIVE   = "EXECUTIVE"
    PRACTITIONER = "PRACTITIONER"
    LEARNER     = "LEARNER"
    ANALYST     = "ANALYST"
    DEFAULT     = "DEFAULT"


@dataclass
class PreferenceVector:
    """
    8-dimensional preference encoding.
    Dimensions: [verbosity, formality, technical_depth, example_density,
                 hedge_tolerance, narrative_vs_list, domain_breadth, response_urgency]
    All values in [0.0, 1.0]
    """
    verbosity: float = 0.5
    formality: float = 0.5
    technical_depth: float = 0.5
    example_density: float = 0.5
    hedge_tolerance: float = 0.5
    narrative_vs_list: float = 0.5
    domain_breadth: float = 0.5
    response_urgency: float = 0.5

    @classmethod
    def default(cls) -> "PreferenceVector":
        return cls()

    def to_list(self) -> list[float]:
        return [
            self.verbosity, self.formality, self.technical_depth,
            self.example_density, self.hedge_tolerance, self.narrative_vs_list,
            self.domain_breadth, self.response_urgency
        ]

    def to_np(self) -> np.ndarray:
        return np.array(self.to_list(), dtype=np.float32)

    def dominant_persona(self) -> str:
        """Fast single-call persona label from dominant dimension."""
        dims = {
            "brief-formal": self.formality * (1 - self.verbosity),
            "deep-technical": self.technical_depth * self.example_density,
            "pedagogical": self.example_density * (1 - self.technical_depth),
            "analytical": self.hedge_tolerance * self.technical_depth,
        }
        label = max(dims, key=dims.get)
        return label

    def top_dimensions(self, n: int = 3) -> dict[str, float]:
        d = asdict(self)
        return dict(sorted(d.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:n])

    def add_laplace_noise(self, epsilon: float = 0.1) -> "PreferenceVector":
        """Differential privacy: Laplace noise for export safety."""
        rng = np.random.default_rng()
        noise = rng.laplace(0, 1/epsilon, 8) * 0.02  # small scale factor
        noisy = np.clip(self.to_np() + noise, 0.0, 1.0)
        fields = list(asdict(self).keys())
        return PreferenceVector(**dict(zip(fields, noisy.tolist())))

    def l2_distance(self, other: "PreferenceVector") -> float:
        return float(np.linalg.norm(self.to_np() - other.to_np()))

    def to_adk_metadata(self) -> dict[str, str]:
        """Google ADK UserContext metadata format (L55 prep)."""
        payload = json.dumps(asdict(self))
        return {
            "preference_vector_b64": base64.b64encode(payload.encode()).decode(),
            "dominant_persona": self.dominant_persona(),
            "schema_version": "1.0"
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreferenceVector":
        fields = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class UserProfile:
    user_id: str
    display_name: str = "Anonymous"
    email: str = ""
    consent_behavioral: bool = False
    consent_embedding: bool = False
    explicit_preferences: dict = field(default_factory=dict)
    preference_vector: Optional[PreferenceVector] = None
    persona: PersonaType = PersonaType.DEFAULT
    interaction_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_adk_user_context(self) -> dict:
        """L55 Google ADK UserContext proto-compatible dict."""
        meta = {}
        if self.preference_vector:
            meta.update(self.preference_vector.to_adk_metadata())
        meta["persona"] = self.persona.value
        meta["interaction_count"] = str(self.interaction_count)
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "metadata": meta
        }


@dataclass
class PersonalizationContext:
    tier: str  # RICH | STANDARD | MINIMAL
    system_prefix: str
    token_cost: int
    persona: PersonaType = PersonaType.DEFAULT
PYEOF

cat > backend/models/interaction.py << 'PYEOF'
from dataclasses import dataclass
from typing import Optional


@dataclass
class Interaction:
    interaction_id: str
    user_id: str
    user_msg: str
    agent_msg: str
    persona_used: str
    context_tier: str
    token_cost: int
    created_at: str
    session_id: str = ""
    feedback_score: Optional[float] = None


@dataclass
class InteractionFeedback:
    interaction_id: str
    score: float   # -1.0 to 1.0 implicit/explicit
    signal_type: str  # "explicit" | "implicit_positive" | "implicit_negative"
PYEOF

cat > backend/models/budget.py << 'PYEOF'
from dataclasses import dataclass


@dataclass
class BudgetState:
    """Imported from L53 BudgetEnforcer — used for context compression decisions."""
    max_tokens: int = 8000
    used_tokens: int = 0
    max_steps: int = 20
    used_steps: int = 0
    max_time_seconds: float = 60.0
    elapsed_seconds: float = 0.0

    def remaining_fraction(self) -> float:
        token_fraction = 1.0 - (self.used_tokens / self.max_tokens)
        step_fraction  = 1.0 - (self.used_steps  / self.max_steps)
        time_fraction  = 1.0 - (self.elapsed_seconds / self.max_time_seconds)
        return min(token_fraction, step_fraction, time_fraction)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)
PYEOF

# =============================================================================
# STEP 3: Database Layer
# =============================================================================
step "Writing database layer"

cat > backend/db/__init__.py << 'PYEOF'
from .store import UserProfileStore
PYEOF

cat > backend/db/store.py << 'PYEOF'
"""
UserProfileStore: SQLite + ChromaDB + Redis hot cache
Designed for portability to Google ADK (L55)
"""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional
import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "data/profiles.db"

CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    display_name TEXT DEFAULT 'Anonymous',
    email TEXT DEFAULT '',
    consent_behavioral INTEGER DEFAULT 0,
    consent_embedding INTEGER DEFAULT 0,
    explicit_preferences TEXT DEFAULT '{}',
    preference_vector TEXT DEFAULT NULL,
    persona TEXT DEFAULT 'DEFAULT',
    interaction_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);
"""

CREATE_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    user_id TEXT,
    user_msg TEXT,
    agent_msg TEXT,
    persona_used TEXT,
    context_tier TEXT,
    token_cost INTEGER DEFAULT 0,
    created_at TEXT,
    session_id TEXT DEFAULT '',
    feedback_score REAL DEFAULT NULL,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);
"""

CREATE_DRIFT_LOG = """
CREATE TABLE IF NOT EXISTS preference_drift_log (
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    l2_distance REAL,
    changed_dimension TEXT,
    delta REAL,
    created_at TEXT
);
"""

CREATE_BUDGET_LOG = """
CREATE TABLE IF NOT EXISTS budget_usage_log (
    log_id TEXT PRIMARY KEY,
    user_id TEXT,
    interaction_id TEXT,
    context_tier TEXT,
    token_cost INTEGER,
    budget_fraction REAL,
    created_at TEXT
);
"""


class UserProfileStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def init_db(self):
        import os
        os.makedirs("data", exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(CREATE_PROFILES)
            await db.execute(CREATE_INTERACTIONS)
            await db.execute(CREATE_DRIFT_LOG)
            await db.execute(CREATE_BUDGET_LOG)
            await db.commit()
        logger.info(f"UserProfileStore initialized: {self.db_path}")

    # ── Profile CRUD ───────────────────────────────────────────────

    async def create_profile(self, user_id: str, display_name: str = "Anonymous",
                              email: str = "", explicit_prefs: dict = None) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        prefs = json.dumps(explicit_prefs or {})
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_profiles
                (user_id, display_name, email, explicit_preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, display_name, email, prefs, now, now))
            await db.commit()
        return await self.get_profile(user_id)

    async def get_profile(self, user_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                d = dict(row)
                d["explicit_preferences"] = json.loads(d["explicit_preferences"] or "{}")
                d["preference_vector"] = json.loads(d["preference_vector"]) if d["preference_vector"] else None
                return d

    async def list_profiles(self, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_profiles ORDER BY updated_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
                result = []
                for row in rows:
                    d = dict(row)
                    d["explicit_preferences"] = json.loads(d["explicit_preferences"] or "{}")
                    d["preference_vector"] = json.loads(d["preference_vector"]) if d["preference_vector"] else None
                    result.append(d)
                return result

    async def update_explicit_preferences(self, user_id: str, prefs: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        # Merge with existing
        existing = await self.get_profile(user_id)
        merged = {}
        if existing:
            merged = existing.get("explicit_preferences", {})
        merged.update(prefs)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE user_profiles
                SET explicit_preferences = ?, updated_at = ?
                WHERE user_id = ?
            """, (json.dumps(merged), now, user_id))
            await db.commit()
        return await self.get_profile(user_id)

    async def update_consent(self, user_id: str, behavioral: bool, embedding: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE user_profiles
                SET consent_behavioral = ?, consent_embedding = ?, updated_at = ?
                WHERE user_id = ?
            """, (int(behavioral), int(embedding), now, user_id))
            await db.commit()

    async def upsert_preference_vector(self, user_id: str, vector: dict,
                                        persona: str = "DEFAULT") -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE user_profiles
                SET preference_vector = ?, persona = ?, updated_at = ?
                WHERE user_id = ?
            """, (json.dumps(vector), persona, now, user_id))
            await db.commit()

    async def delete_profile(self, user_id: str) -> bool:
        """Right-to-forget: cascade delete all user data."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM preference_drift_log WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM budget_usage_log WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            await db.commit()
        logger.info(f"Profile deleted (right-to-forget): {user_id}")
        return True

    # ── Interactions ────────────────────────────────────────────────

    async def save_interaction(self, user_id: str, user_msg: str, agent_msg: str,
                                persona: str, context_tier: str,
                                token_cost: int = 0, session_id: str = "") -> str:
        interaction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO interactions
                (interaction_id, user_id, user_msg, agent_msg, persona_used,
                 context_tier, token_cost, created_at, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (interaction_id, user_id, user_msg, agent_msg, persona,
                  context_tier, token_cost, now, session_id))
            await db.execute("""
                UPDATE user_profiles
                SET interaction_count = interaction_count + 1, updated_at = ?
                WHERE user_id = ?
            """, (now, user_id))
            await db.commit()
        return interaction_id

    async def get_recent_interactions(self, user_id: str, limit: int = 20) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM interactions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit)) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def submit_feedback(self, interaction_id: str, score: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE interactions SET feedback_score = ?
                WHERE interaction_id = ?
            """, (score, interaction_id))
            await db.commit()

    # ── Drift Log ────────────────────────────────────────────────────

    async def log_preference_drift(self, user_id: str, l2_distance: float,
                                    changed_dim: str, delta: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO preference_drift_log
                (log_id, user_id, l2_distance, changed_dimension, delta, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), user_id, l2_distance, changed_dim, delta, now))
            await db.commit()

    async def get_drift_history(self, user_id: str, limit: int = 30) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM preference_drift_log
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
            """, (user_id, limit)) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── Stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM user_profiles") as c:
                profiles = (await c.fetchone())[0]
            async with db.execute("SELECT COUNT(*) FROM interactions") as c:
                interactions = (await c.fetchone())[0]
            async with db.execute(
                "SELECT persona, COUNT(*) as cnt FROM user_profiles GROUP BY persona"
            ) as c:
                persona_dist = {r[0]: r[1] for r in await c.fetchall()}
            async with db.execute(
                "SELECT context_tier, COUNT(*) as cnt FROM interactions GROUP BY context_tier"
            ) as c:
                tier_dist = {r[0]: r[1] for r in await c.fetchall()}
        return {
            "total_profiles": profiles,
            "total_interactions": interactions,
            "persona_distribution": persona_dist,
            "context_tier_distribution": tier_dist
        }
PYEOF

# =============================================================================
# STEP 4: Services
# =============================================================================
step "Writing core services"

cat > backend/services/__init__.py << 'PYEOF'
from .inference import PreferenceInferenceEngine
from .router import AdaptiveAgentRouter
from .compressor import ContextCompressor
PYEOF

cat > backend/services/inference.py << 'PYEOF'
"""
PreferenceInferenceEngine — uses Gemini 2.0 Flash to derive behavioral
preference vectors from raw interaction history.
"""
from __future__ import annotations
import json
import logging
import numpy as np
from typing import Optional
import google.generativeai as genai

from ..models.profile import PreferenceVector
from ..db.store import UserProfileStore
from .router import AdaptiveAgentRouter

logger = logging.getLogger(__name__)

INFERENCE_PROMPT = """You are an AI behavioral analyst. Analyze the following conversation history between a user and an AI agent. Infer the user's communication style preferences.

CONVERSATION HISTORY:
{corpus}

Analyze patterns in:
- How long the user's messages are (verbosity)
- Formality of language
- How technical the topics they pursue are
- Whether they ask for examples
- How comfortable they are with uncertainty ("maybe", "approximately")
- Whether they prefer narrative explanations or bullet lists
- How broad vs narrow their topics are
- How urgent or patient they seem

Return ONLY valid JSON (no markdown, no preamble) with this exact structure:
{{
  "verbosity": <0.0-1.0>,
  "formality": <0.0-1.0>,
  "technical_depth": <0.0-1.0>,
  "example_density": <0.0-1.0>,
  "hedge_tolerance": <0.0-1.0>,
  "narrative_vs_list": <0.0-1.0>,
  "domain_breadth": <0.0-1.0>,
  "response_urgency": <0.0-1.0>,
  "reasoning": "<one sentence rationale>"
}}"""

DRIFT_THRESHOLD = 0.3  # L2 distance trigger


class PreferenceInferenceEngine:
    def __init__(self, store: UserProfileStore, api_key: str,
                 privacy_mode: bool = True):
        self.store = store
        self.privacy_mode = privacy_mode
        self.router = AdaptiveAgentRouter()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def infer_and_store(self, user_id: str) -> dict:
        """Full inference pipeline: history → vector → persona → store → drift check."""
        profile = await self.store.get_profile(user_id)
        if not profile:
            return {"error": "Profile not found", "user_id": user_id}

        if not profile.get("consent_behavioral"):
            return {"error": "User has not consented to behavioral tracking", "user_id": user_id}

        history = await self.store.get_recent_interactions(user_id, limit=20)
        if not history:
            return {"status": "insufficient_data", "message": "Need at least 1 interaction", "user_id": user_id}

        # Build corpus
        corpus_lines = []
        for h in reversed(history):  # chronological
            corpus_lines.append(f"USER: {h['user_msg'][:300]}")
            corpus_lines.append(f"AGENT: {h['agent_msg'][:300]}")
        corpus = "\n".join(corpus_lines)

        # Gemini inference
        prompt = INFERENCE_PROMPT.format(corpus=corpus)
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            # Strip possible markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw = json.loads(raw_text)
            reasoning = raw.pop("reasoning", "")
            new_vector = PreferenceVector.from_dict(raw)
        except Exception as e:
            logger.warning("Inference unavailable for %s (%s); using fallback vector", user_id, e)
            merged = {**PreferenceVector.default().__dict__, **profile.get("explicit_preferences", {})}
            new_vector = PreferenceVector.from_dict(merged)
            reasoning = "Fallback: Gemini unavailable or quota-limited; derived from explicit preferences."
            fallback_persona = self.router.select_persona(new_vector)

            await self.store.upsert_preference_vector(
                user_id,
                vars(new_vector.add_laplace_noise(epsilon=0.1) if self.privacy_mode else new_vector),
                fallback_persona.value,
            )
            return {
                "status": "fallback",
                "user_id": user_id,
                "persona": fallback_persona.value,
                "preference_vector": vars(new_vector),
                "reasoning": reasoning,
                "interactions_analyzed": len(history),
                "privacy_noise_applied": self.privacy_mode,
                "message": "Gemini quota exceeded/unavailable. Using fallback inference."
            }

        # Apply differential privacy for export safety
        export_vector = new_vector.add_laplace_noise(epsilon=0.1) if self.privacy_mode else new_vector

        # Drift detection (L54 assignment pattern)
        drift_result = None
        old_vec_data = profile.get("preference_vector")
        if old_vec_data:
            old_vector = PreferenceVector.from_dict(old_vec_data)
            l2 = new_vector.l2_distance(old_vector)
            if l2 > DRIFT_THRESHOLD:
                # Find largest changed dimension
                old_arr = np.array(old_vector.to_list())
                new_arr = np.array(new_vector.to_list())
                deltas = np.abs(new_arr - old_arr)
                dim_names = list(PreferenceVector.__dataclass_fields__.keys())
                max_idx = int(np.argmax(deltas))
                changed_dim = dim_names[max_idx]
                delta = float(deltas[max_idx])
                await self.store.log_preference_drift(user_id, l2, changed_dim, delta)
                drift_result = {
                    "drift_detected": True,
                    "l2_distance": round(l2, 4),
                    "changed_dimension": changed_dim,
                    "delta": round(delta, 4)
                }

        # Select persona
        persona = self.router.select_persona(new_vector)

        # Store
        await self.store.upsert_preference_vector(
            user_id, vars(export_vector), persona.value
        )

        result = {
            "status": "success",
            "user_id": user_id,
            "persona": persona.value,
            "preference_vector": vars(new_vector),
            "reasoning": reasoning,
            "interactions_analyzed": len(history),
            "privacy_noise_applied": self.privacy_mode
        }
        if drift_result:
            result["drift"] = drift_result

        logger.info(f"Inference complete for {user_id}: persona={persona.value}")
        return result
PYEOF

cat > backend/services/router.py << 'PYEOF'
"""
AdaptiveAgentRouter — deterministic persona selection via cosine similarity.
No LLM calls in the routing path.
"""
from __future__ import annotations
import numpy as np
from ..models.profile import PersonaType, PreferenceVector

# 8-dim persona archetype vectors
# [verbosity, formality, technical_depth, example_density,
#  hedge_tolerance, narrative_vs_list, domain_breadth, response_urgency]
PERSONA_ARCHETYPES = {
    PersonaType.EXECUTIVE:    np.array([0.2, 0.9, 0.4, 0.2, 0.2, 0.9, 0.5, 0.9], dtype=np.float32),
    PersonaType.PRACTITIONER: np.array([0.6, 0.4, 0.9, 0.9, 0.2, 0.3, 0.7, 0.6], dtype=np.float32),
    PersonaType.LEARNER:      np.array([0.7, 0.5, 0.4, 0.9, 0.7, 0.2, 0.4, 0.3], dtype=np.float32),
    PersonaType.ANALYST:      np.array([0.6, 0.7, 0.8, 0.5, 0.9, 0.5, 0.8, 0.5], dtype=np.float32),
}

PERSONA_SYSTEM_PROMPTS = {
    PersonaType.EXECUTIVE: (
        "You are a concise, high-impact advisor. "
        "Lead with the bottom line. Use bullets. No hedging. Max 3 sentences per point. "
        "Assume C-suite context."
    ),
    PersonaType.PRACTITIONER: (
        "You are a hands-on technical expert. "
        "Be precise, include concrete code examples, skip basic explanations. "
        "Use production-grade terminology."
    ),
    PersonaType.LEARNER: (
        "You are a patient, encouraging teacher. "
        "Build up from first principles. Use analogies and worked examples. "
        "Check understanding at the end."
    ),
    PersonaType.ANALYST: (
        "You are a rigorous analyst. "
        "Quantify claims where possible. Flag uncertainty explicitly. "
        "Structure responses with clear reasoning chains."
    ),
    PersonaType.DEFAULT: (
        "You are a helpful, balanced AI assistant."
    ),
}


class AdaptiveAgentRouter:
    def select_persona(self, vector: PreferenceVector) -> PersonaType:
        """Cosine similarity against archetype vectors. O(1), no LLM."""
        v = vector.to_np()
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return PersonaType.DEFAULT

        scores = {}
        for ptype, archetype in PERSONA_ARCHETYPES.items():
            similarity = float(np.dot(v, archetype) / (norm_v * np.linalg.norm(archetype)))
            scores[ptype] = similarity

        return max(scores, key=scores.get)

    def get_system_prompt(self, persona: PersonaType) -> str:
        return PERSONA_SYSTEM_PROMPTS.get(persona, PERSONA_SYSTEM_PROMPTS[PersonaType.DEFAULT])

    def score_all(self, vector: PreferenceVector) -> dict[str, float]:
        v = vector.to_np()
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return {p.value: 0.25 for p in PersonaType if p != PersonaType.DEFAULT}
        return {
            ptype.value: round(float(
                np.dot(v, archetype) / (norm_v * np.linalg.norm(archetype))
            ), 4)
            for ptype, archetype in PERSONA_ARCHETYPES.items()
        }
PYEOF

cat > backend/services/compressor.py << 'PYEOF'
"""
ContextCompressor — budget-aware personalization context builder.
Integrates with L53 BudgetEnforcer semantics.
"""
from __future__ import annotations
from ..models.profile import PreferenceVector, PersonalizationContext, PersonaType
from ..models.budget import BudgetState
from .router import AdaptiveAgentRouter

_router = AdaptiveAgentRouter()


def _rich_prefix(vector: PreferenceVector, persona: PersonaType) -> str:
    top = vector.top_dimensions(n=3)
    dims_desc = ", ".join([f"{k.replace('_',' ')}={v:.2f}" for k, v in top.items()])
    system = _router.get_system_prompt(persona)
    return (
        f"{system}\n\n"
        f"[User calibration — top preferences: {dims_desc}. "
        f"Adjust depth, formality, and format accordingly.]"
    )


def _standard_prefix(vector: PreferenceVector, persona: PersonaType) -> str:
    system = _router.get_system_prompt(persona)
    top = vector.top_dimensions(n=1)
    key, val = list(top.items())[0]
    hint = "high" if val > 0.7 else ("low" if val < 0.3 else "moderate")
    return f"{system}\n[User preference: {hint} {key.replace('_',' ')}.]"


def _minimal_prefix(persona: PersonaType) -> str:
    return f"[Persona: {persona.value}] {_router.get_system_prompt(persona)[:80]}"


class ContextCompressor:
    """
    Three-tier compression based on remaining budget fraction.
    Integrates L53 BudgetState for context-aware degradation.
    """

    def build_context(self, vector: PreferenceVector,
                      budget: BudgetState,
                      persona: PersonaType) -> PersonalizationContext:
        fraction = budget.remaining_fraction()

        if fraction > 0.6:
            prefix = _rich_prefix(vector, persona)
            return PersonalizationContext(
                tier="RICH", system_prefix=prefix, token_cost=85, persona=persona
            )
        elif fraction > 0.3:
            prefix = _standard_prefix(vector, persona)
            return PersonalizationContext(
                tier="STANDARD", system_prefix=prefix, token_cost=32, persona=persona
            )
        else:
            prefix = _minimal_prefix(persona)
            return PersonalizationContext(
                tier="MINIMAL", system_prefix=prefix, token_cost=8, persona=persona
            )

    def build_default(self, persona: PersonaType = PersonaType.DEFAULT) -> PersonalizationContext:
        budget = BudgetState()  # full budget
        vector = PreferenceVector.default()
        return self.build_context(vector, budget, persona)
PYEOF

# =============================================================================
# STEP 5: Routers
# =============================================================================
step "Writing API routers"

cat > backend/routers/__init__.py << 'PYEOF'
from .profiles import router as profiles_router
from .chat import router as chat_router
from .analytics import router as analytics_router
PYEOF

cat > backend/routers/profiles.py << 'PYEOF'
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
PYEOF

cat > backend/routers/chat.py << 'PYEOF'
from fastapi import APIRouter, Body, Header
from pydantic import BaseModel
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
PYEOF

cat > backend/routers/analytics.py << 'PYEOF'
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
PYEOF

# =============================================================================
# STEP 6: Config + Main App
# =============================================================================
step "Writing config and main app"

cat > backend/config.py << 'PYEOF'
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "data/profiles.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PRIVACY_MODE = os.getenv("PRIVACY_MODE", "true").lower() == "true"
PYEOF

cat > backend/main.py << 'PYEOF'
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
PYEOF

# =============================================================================
# STEP 7: React Frontend
# =============================================================================
step "Writing React frontend"

cat > frontend/package.json << 'JSON'
{
  "name": "l54-personalization-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.12.7",
    "lucide-react": "^0.400.0"
  },
  "devDependencies": {
    "vite": "^5.3.4",
    "@vitejs/plugin-react": "^4.3.1"
  },
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }
}
JSON

cat > frontend/vite.config.js << 'JS'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: { port: 3054, proxy: { '/api': { target: 'http://localhost:8054', rewrite: p => p.replace(/^\/api/, '') } } }
})
JS

cat > frontend/index.html << 'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>L54 · Agent Personalization</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
HTML

cat > frontend/src/main.jsx << 'JSX'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>
)
JSX

cat > frontend/src/index.css << 'CSS'
:root {
  --bg: #0e0f14;
  --surface: #161720;
  --surface2: #1e2030;
  --border: #2a2d3e;
  --accent: #6ee7b7;
  --accent2: #f59e0b;
  --accent3: #818cf8;
  --text: #e2e8f0;
  --text-muted: #64748b;
  --executive: #f59e0b;
  --practitioner: #6ee7b7;
  --learner: #818cf8;
  --analyst: #fb7185;
  --font-mono: 'DM Mono', monospace;
  --font-display: 'Syne', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font-mono); }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
CSS

cat > frontend/src/App.jsx << 'JSX'
import React, { useState, useEffect, useRef } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar, Legend, Cell
} from 'recharts'

const API = ''  // proxied via vite

const PERSONA_COLORS = {
  EXECUTIVE: '#f59e0b',
  PRACTITIONER: '#6ee7b7',
  LEARNER: '#818cf8',
  ANALYST: '#fb7185',
  DEFAULT: '#64748b'
}

const PERSONA_ICONS = {
  EXECUTIVE: '⚡',
  PRACTITIONER: '⚙️',
  LEARNER: '📚',
  ANALYST: '📊',
  DEFAULT: '🤖'
}

// ── Utility ──────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts
  })
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`)
  return r.json()
}

// ── Header ───────────────────────────────────────────────────────
function Header({ activeTab, setActiveTab }) {
  const tabs = ['profiles', 'chat', 'compare', 'analytics']
  return (
    <header style={{
      borderBottom: '1px solid var(--border)',
      padding: '0 24px',
      display: 'flex', alignItems: 'center', gap: 32, height: 56
    }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 18, color: 'var(--accent)' }}>
        L54<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> · Personalization</span>
      </div>
      <nav style={{ display: 'flex', gap: 4 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            background: activeTab === t ? 'var(--surface2)' : 'transparent',
            border: activeTab === t ? '1px solid var(--border)' : '1px solid transparent',
            color: activeTab === t ? 'var(--text)' : 'var(--text-muted)',
            padding: '6px 14px', borderRadius: 6, cursor: 'pointer',
            fontFamily: 'var(--font-mono)', fontSize: 13, textTransform: 'capitalize',
            transition: 'all 0.15s'
          }}>
            {t}
          </button>
        ))}
      </nav>
    </header>
  )
}

// ── PersonaBadge ─────────────────────────────────────────────────
function PersonaBadge({ persona, size = 'sm' }) {
  const color = PERSONA_COLORS[persona] || PERSONA_COLORS.DEFAULT
  const icon = PERSONA_ICONS[persona] || '🤖'
  const fs = size === 'lg' ? 15 : 12
  return (
    <span style={{
      background: `${color}18`, border: `1px solid ${color}55`,
      color, borderRadius: 20, padding: size === 'lg' ? '4px 12px' : '2px 8px',
      fontSize: fs, fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 4
    }}>
      {icon} {persona}
    </span>
  )
}

// ── PreferenceRadar ───────────────────────────────────────────────
function PreferenceRadar({ vector }) {
  if (!vector) return <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No preference data yet</div>
  const data = Object.entries(vector).map(([key, val]) => ({
    dim: key.replace(/_/g, ' ').replace('narrative vs list', 'narrative'),
    value: Math.round(val * 100)
  }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="dim" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <Radar dataKey="value" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.18} strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

// ── ProfileCard ───────────────────────────────────────────────────
function ProfileCard({ profile, onSelect, selected }) {
  const persona = profile.persona || 'DEFAULT'
  const color = PERSONA_COLORS[persona]
  return (
    <div onClick={() => onSelect(profile)} style={{
      background: 'var(--surface)',
      border: `1px solid ${selected ? color : 'var(--border)'}`,
      borderRadius: 10, padding: '14px 16px', cursor: 'pointer',
      transition: 'all 0.15s',
      boxShadow: selected ? `0 0 0 1px ${color}44` : 'none'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 14, marginBottom: 2 }}>
            {profile.display_name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{profile.user_id}</div>
        </div>
        <PersonaBadge persona={persona} />
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', gap: 12 }}>
        <span>💬 {profile.interaction_count} chats</span>
        <span>{profile.consent_behavioral ? '🔍 tracking on' : '🔒 no tracking'}</span>
      </div>
    </div>
  )
}

// ── ProfileDetail ─────────────────────────────────────────────────
function ProfileDetail({ profile, onRefresh }) {
  const [inferring, setInferring] = useState(false)
  const [inferResult, setInferResult] = useState(null)
  const [personaScores, setPersonaScores] = useState(null)

  useEffect(() => {
    if (!profile) return
    apiFetch(`/profiles/${profile.user_id}/persona`)
      .then(r => setPersonaScores(r.scores))
      .catch(() => {})
  }, [profile])

  const handleInfer = async () => {
    setInferring(true)
    try {
      const r = await apiFetch(`/profiles/${profile.user_id}/infer`, { method: 'POST' })
      setInferResult(r)
      onRefresh()
    } catch (e) {
      setInferResult({ error: e.message })
    }
    setInferring(false)
  }

  if (!profile) return null

  const scoreData = personaScores ? Object.entries(personaScores).map(([p, s]) => ({
    persona: p, score: Math.round(s * 100)
  })) : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
          Preference Vector
        </div>
        <PreferenceRadar vector={profile.preference_vector} />
        <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
          <button onClick={handleInfer} disabled={inferring || !profile.consent_behavioral} style={{
            background: inferring ? 'var(--surface2)' : 'var(--accent)', color: 'var(--bg)',
            border: 'none', borderRadius: 6, padding: '7px 14px', cursor: inferring ? 'not-allowed' : 'pointer',
            fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)',
            opacity: !profile.consent_behavioral ? 0.4 : 1
          }}>
            {inferring ? '⟳ Inferring...' : '⚡ Run Inference'}
          </button>
          {!profile.consent_behavioral && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>
              Enable behavioral consent first
            </span>
          )}
        </div>
        {inferResult && (
          <div style={{
            marginTop: 10, fontSize: 11, padding: '8px 10px',
            background: inferResult.error ? '#fb718518' : '#6ee7b718',
            borderRadius: 6, color: inferResult.error ? '#fb7185' : 'var(--accent)'
          }}>
            {inferResult.error ? `✗ ${inferResult.error}` : `✓ ${inferResult.persona} · ${inferResult.interactions_analyzed} interactions analyzed`}
            {inferResult.drift && (
              <div style={{ marginTop: 4, color: '#f59e0b' }}>
                ⚠ Drift detected: {inferResult.drift.changed_dimension} Δ{inferResult.drift.delta.toFixed(2)}
              </div>
            )}
          </div>
        )}
      </div>
      <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
          Persona Scores
        </div>
        {scoreData.length > 0 ? (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={scoreData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} domain={[0, 100]} />
              <YAxis type="category" dataKey="persona" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }}
                formatter={v => [`${v}%`, 'Similarity']}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                {scoreData.map((entry, i) => (
                  <Cell key={i} fill={PERSONA_COLORS[entry.persona] || PERSONA_COLORS.DEFAULT} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 20 }}>
            Run inference to see persona scores
          </div>
        )}
        <div style={{ marginTop: 12, fontSize: 12 }}>
          <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>Explicit preferences:</div>
          {Object.entries(profile.explicit_preferences || {}).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--text-muted)' }}>{k}</span>
              <span style={{ color: 'var(--accent2)' }}>{typeof v === 'number' ? v.toFixed(2) : v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Profiles Tab ──────────────────────────────────────────────────
function ProfilesTab() {
  const [profiles, setProfiles] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)

  const loadProfiles = async () => {
    try {
      const data = await apiFetch('/profiles')
      setProfiles(data)
      if (data.length > 0 && !selected) setSelected(data[0])
    } catch (e) {
      console.error(e)
    }
    setLoading(false)
  }

  useEffect(() => { loadProfiles() }, [])

  return (
    <div style={{ padding: 24, display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20, height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      <div style={{ overflow: 'auto' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 16, marginBottom: 14, color: 'var(--text-muted)' }}>
          User Profiles ({profiles.length})
        </div>
        {loading ? <div style={{ color: 'var(--text-muted)' }}>Loading...</div> : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {profiles.map(p => (
              <ProfileCard key={p.user_id} profile={p} selected={selected?.user_id === p.user_id} onSelect={setSelected} />
            ))}
          </div>
        )}
      </div>
      <div style={{ overflow: 'auto' }}>
        {selected && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 22 }}>
                {selected.display_name}
              </div>
              <PersonaBadge persona={selected.persona || 'DEFAULT'} size="lg" />
            </div>
            <ProfileDetail
              profile={profiles.find(p => p.user_id === selected.user_id)}
              onRefresh={loadProfiles}
            />
          </>
        )}
      </div>
    </div>
  )
}

// ── Chat Tab ──────────────────────────────────────────────────────
function ChatTab() {
  const [profiles, setProfiles] = useState([])
  const [userId, setUserId] = useState('exec-001')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [budget, setBudget] = useState(1.0)
  const endRef = useRef(null)
  const demoQueries = [
    "Give me a 5-bullet executive summary of this week's platform risks.",
    'Explain distributed tracing with a concrete OpenTelemetry example.',
    'Teach me Kubernetes in simple terms with one analogy.',
    'Compare Postgres vs MongoDB for analytics workload with tradeoffs.',
    'Create a 30-day learning plan for becoming production-ready in SRE.'
  ]

  useEffect(() => {
    apiFetch('/profiles').then(setProfiles).catch(() => {})
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (messageOverride = null) => {
    const msg = (messageOverride ?? input).trim()
    if (!msg || loading) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: msg }])
    setLoading(true)
    try {
      const r = await apiFetch('/chat', {
        method: 'POST',
        headers: { 'X-User-Id': userId },
        body: JSON.stringify({ message: msg, budget_fraction: budget })
      })
      setMessages(m => [...m, {
        role: 'agent', content: r.response, persona: r.persona,
        tier: r.context_tier, interactionId: r.interaction_id
      }])
    } catch (e) {
      setMessages(m => [...m, { role: 'error', content: e.message }])
    }
    setLoading(false)
  }

  const handleFeedback = async (id, score) => {
    await apiFetch('/chat/feedback', {
      method: 'POST', body: JSON.stringify({ interaction_id: id, score })
    })
  }

  const profile = profiles.find(p => p.user_id === userId)
  const persona = profile?.persona || 'DEFAULT'

  return (
    <div style={{ height: 'calc(100vh - 56px)', display: 'flex', flexDirection: 'column', padding: 24, gap: 16 }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>User</label>
          <select value={userId} onChange={e => { setUserId(e.target.value); setMessages([]) }} style={{
            background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)',
            borderRadius: 6, padding: '6px 10px', fontSize: 13, fontFamily: 'var(--font-mono)'
          }}>
            {profiles.map(p => <option key={p.user_id} value={p.user_id}>{p.display_name}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
            Budget fraction (L53): {budget.toFixed(1)}
          </label>
          <input type="range" min="0.1" max="1.0" step="0.1" value={budget}
            onChange={e => setBudget(parseFloat(e.target.value))}
            style={{ width: 160 }}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>Active Persona</label>
          <PersonaBadge persona={persona} size="lg" />
        </div>
      </div>

      {/* Demo Queries */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {demoQueries.map((q, i) => (
          <button key={i} onClick={() => send(q)} disabled={loading} style={{
            background: 'var(--surface2)', border: '1px solid var(--border)',
            color: 'var(--text-muted)', borderRadius: 999, padding: '6px 10px',
            cursor: loading ? 'not-allowed' : 'pointer', fontSize: 12, fontFamily: 'var(--font-mono)',
            opacity: loading ? 0.5 : 1
          }}>
            {q.length > 58 ? `${q.slice(0, 58)}...` : q}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 60, fontSize: 13 }}>
            Start chatting — responses adapt to {profile?.display_name || 'user'}'s profile
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '72%'
          }}>
            {m.role === 'user' ? (
              <div style={{
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: '12px 12px 2px 12px', padding: '10px 14px', fontSize: 14
              }}>
                {m.content}
              </div>
            ) : m.role === 'agent' ? (
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, display: 'flex', gap: 8 }}>
                  <PersonaBadge persona={m.persona} />
                  <span style={{
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 10, padding: '1px 7px', fontSize: 11
                  }}>
                    {m.tier}
                  </span>
                </div>
                <div style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: '2px 12px 12px 12px', padding: '10px 14px', fontSize: 14,
                  lineHeight: 1.6, whiteSpace: 'pre-wrap'
                }}>
                  {m.content}
                </div>
                <div style={{ marginTop: 6, display: 'flex', gap: 6 }}>
                  <button onClick={() => handleFeedback(m.interactionId, 1.0)} style={{
                    background: 'transparent', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 12,
                    color: 'var(--text-muted)'
                  }}>👍</button>
                  <button onClick={() => handleFeedback(m.interactionId, -1.0)} style={{
                    background: 'transparent', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '3px 8px', cursor: 'pointer', fontSize: 12,
                    color: 'var(--text-muted)'
                  }}>👎</button>
                </div>
              </div>
            ) : (
              <div style={{ color: '#fb7185', fontSize: 13 }}>Error: {m.content}</div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', color: 'var(--text-muted)', fontSize: 13 }}>
            <span style={{ animation: 'pulse 1s infinite' }}>● generating...</span>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask anything — response adapts to persona..."
          style={{
            flex: 1, background: 'var(--surface2)', border: '1px solid var(--border)',
            borderRadius: 8, padding: '10px 14px', color: 'var(--text)', fontSize: 14,
            fontFamily: 'var(--font-mono)', outline: 'none'
          }}
        />
        <button onClick={send} disabled={loading} style={{
          background: 'var(--accent)', color: 'var(--bg)', border: 'none',
          borderRadius: 8, padding: '10px 20px', cursor: 'pointer',
          fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: 14,
          opacity: loading ? 0.5 : 1
        }}>Send</button>
      </div>
    </div>
  )
}

// ── Compare Tab ───────────────────────────────────────────────────
function CompareTab() {
  const [profiles, setProfiles] = useState([])
  const [selectedIds, setSelectedIds] = useState([])
  const [message, setMessage] = useState('Explain distributed tracing in a microservices architecture.')
  const [results, setResults] = useState([])
  const [running, setRunning] = useState(false)

  useEffect(() => {
    apiFetch('/profiles').then(d => {
      setProfiles(d)
      if (d.length >= 2) setSelectedIds(d.slice(0, 2).map(p => p.user_id))
    }).catch(() => {})
  }, [])

  const toggle = (id) => {
    setSelectedIds(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id].slice(-4))
  }

  const run = async () => {
    if (selectedIds.length < 1) return
    setRunning(true); setResults([])
    try {
      const r = await apiFetch('/chat/compare', {
        method: 'POST',
        body: JSON.stringify({ message, user_ids: selectedIds })
      })
      setResults(r.comparisons || [])
    } catch (e) { console.error(e) }
    setRunning(false)
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18, marginBottom: 16 }}>
        Persona A/B Comparison
      </div>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
          Select users (max 4)
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {profiles.map(p => (
            <button key={p.user_id} onClick={() => toggle(p.user_id)} style={{
              background: selectedIds.includes(p.user_id) ? `${PERSONA_COLORS[p.persona || 'DEFAULT']}22` : 'var(--surface)',
              border: `1px solid ${selectedIds.includes(p.user_id) ? PERSONA_COLORS[p.persona || 'DEFAULT'] : 'var(--border)'}`,
              color: 'var(--text)', borderRadius: 8, padding: '6px 14px',
              cursor: 'pointer', fontSize: 13, fontFamily: 'var(--font-mono)'
            }}>
              {p.display_name}
            </button>
          ))}
        </div>
      </div>
      <div style={{ marginBottom: 14 }}>
        <textarea value={message} onChange={e => setMessage(e.target.value)} style={{
          width: '100%', maxWidth: 700, background: 'var(--surface2)',
          border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px',
          color: 'var(--text)', fontFamily: 'var(--font-mono)', fontSize: 14,
          resize: 'vertical', rows: 3, minHeight: 80
        }} />
      </div>
      <button onClick={run} disabled={running || selectedIds.length < 1} style={{
        background: 'var(--accent2)', color: 'var(--bg)', border: 'none',
        borderRadius: 8, padding: '8px 20px', cursor: 'pointer',
        fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14,
        marginBottom: 20, opacity: running ? 0.5 : 1
      }}>
        {running ? '⟳ Running...' : '▶ Compare Responses'}
      </button>
      {results.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
          {results.map((r, i) => (
            <div key={i} style={{ background: 'var(--surface)', border: `1px solid ${PERSONA_COLORS[r.persona]}55`, borderRadius: 10, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 14 }}>{r.user_id}</span>
                <PersonaBadge persona={r.persona} />
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
                {r.response}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Analytics Tab ─────────────────────────────────────────────────
function AnalyticsTab() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    const load = () => apiFetch('/analytics/stats').then(setStats).catch(() => {})
    load()
    const id = setInterval(load, 4000)
    return () => clearInterval(id)
  }, [])

  if (!stats) return <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading...</div>

  const personaData = Object.entries(stats.persona_distribution || {}).map(([k, v]) => ({ persona: k, count: v }))
  const tierData = Object.entries(stats.context_tier_distribution || {}).map(([k, v]) => ({ tier: k, count: v }))

  return (
    <div style={{ padding: 24 }}>
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18, marginBottom: 20 }}>Analytics</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Profiles', val: stats.total_profiles, color: 'var(--accent)' },
          { label: 'Total Interactions', val: stats.total_interactions, color: 'var(--accent2)' },
          { label: 'Personas Active', val: Object.keys(stats.persona_distribution || {}).length, color: 'var(--accent3)' },
        ].map(m => (
          <div key={m.label} style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: '16px 20px' }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{m.label}</div>
            <div style={{ fontSize: 36, fontFamily: 'var(--font-display)', fontWeight: 800, color: m.color, marginTop: 4 }}>{m.val}</div>
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Persona Distribution</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={personaData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="persona" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }} />
              <Bar dataKey="count" fill="var(--accent3)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ background: 'var(--surface)', borderRadius: 10, border: '1px solid var(--border)', padding: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Context Tier Usage (L53 Budget Integration)</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={tierData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="tier" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', fontSize: 12 }} />
              <Bar dataKey="count" fill="var(--accent2)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('profiles')
  const tabs = { profiles: <ProfilesTab />, chat: <ChatTab />, compare: <CompareTab />, analytics: <AnalyticsTab /> }
  return (
    <div style={{ minHeight: '100vh' }}>
      <Header activeTab={tab} setActiveTab={setTab} />
      {tabs[tab]}
    </div>
  )
}
JSX

# =============================================================================
# STEP 8: Docker + Helper Scripts
# =============================================================================
step "Writing Docker and helper scripts"

cat > docker/docker-compose.yml << 'YAML'
version: "3.9"
services:
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8054:8054"
    volumes:
      - ../data:/app/data
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}
      - PRIVACY_MODE=true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8054/health"]
      interval: 15s
      timeout: 5s
      retries: 3

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3054:80"
    depends_on:
      backend:
        condition: service_healthy
YAML

cat > docker/Dockerfile.backend << 'DOCKER'
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY data/ ./data/ 2>/dev/null || true
RUN mkdir -p data
EXPOSE 8054
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8054"]
DOCKER

cat > docker/Dockerfile.frontend << 'DOCKER'
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json .
RUN npm install
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
DOCKER

cat > docker/nginx.conf << 'NGINX'
server {
  listen 80;
  root /usr/share/nginx/html;
  index index.html;
  location / { try_files $uri $uri/ /index.html; }
  location /api/ {
    proxy_pass http://backend:8054/;
    proxy_set_header Host $host;
  }
}
NGINX

cat > backend/requirements.txt << 'TXT'
fastapi==0.115.0
uvicorn[standard]==0.30.6
aiosqlite==0.20.0
google-generativeai==0.8.3
pydantic==2.9.2
numpy==2.1.1
python-multipart==0.0.9
httpx==0.27.2
apscheduler==3.10.4
TXT

# ── build.sh ──────────────────────────────────────────────────────
cat > scripts/build.sh << 'SH'
#!/usr/bin/env bash
set -e
echo "[L54] Building..."
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install -q -r backend/requirements.txt
cd frontend && npm install --silent && cd ..
echo "[L54] Build complete"
SH
chmod +x scripts/build.sh

# ── start.sh ─────────────────────────────────────────────────────
cat > scripts/start.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d "$ROOT/.venv" ] || { echo "[L54] Missing .venv — run scripts/build.sh first"; exit 1; }
[ -f "$ROOT/scripts/stop.sh" ] && bash "$ROOT/scripts/stop.sh" || true
mkdir -p data logs
source "$ROOT/.venv/bin/activate"

echo "[L54] Starting backend on :8054..."
uvicorn backend.main:app --host 0.0.0.0 --port 8054 --reload \
  > "$ROOT/logs/backend.log" 2>&1 &
echo $! > "$ROOT/.backend.pid"

sleep 2

echo "[L54] Starting frontend on :3054..."
cd "$ROOT/frontend"
npm run dev > "$ROOT/logs/frontend.log" 2>&1 &
echo $! > "$ROOT/.frontend.pid"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  L54 Agent Personalization           ║"
echo "║  Backend  → http://localhost:8054    ║"
echo "║  Frontend → http://localhost:3054    ║"
echo "║  API Docs → http://localhost:8054/docs ║"
echo "╚══════════════════════════════════════╝"
SH
chmod +x scripts/start.sh

# ── stop.sh ──────────────────────────────────────────────────────
cat > scripts/stop.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
kill_pidfile() {
  local f="$1"
  if [ -f "$f" ]; then
    local pid
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    rm -f "$f"
  fi
}
kill_pidfile "$ROOT/.backend.pid"
kill_pidfile "$ROOT/.frontend.pid"
# Clear duplicate listeners on lesson ports (stale PID files)
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8054/tcp >/dev/null 2>&1 || true
  fuser -k 3054/tcp >/dev/null 2>&1 || true
fi
echo "[L54] Stopped"
SH
chmod +x scripts/stop.sh

# ── test.sh ──────────────────────────────────────────────────────
cat > scripts/test.sh << 'SH'
#!/usr/bin/env bash
BASE="http://localhost:8054"
PASS=0; FAIL=0

check() {
  local label="$1"; local cmd="$2"; local expect="$3"
  result=$(eval "$cmd" 2>/dev/null)
  if echo "$result" | grep -q "$expect"; then
    echo -e "\033[0;32m✓\033[0m $label"
    ((PASS++))
  else
    echo -e "\033[0;31m✗\033[0m $label → expected '$expect'"
    echo "  Got: $(echo $result | head -c 120)"
    ((FAIL++))
  fi
}

echo "=== L54 Test Suite ==="

check "Health check" "curl -sf $BASE/health" '"ok"'
check "List profiles" "curl -sf $BASE/profiles" '"exec-001"'

check "Create test profile" \
  "curl -sf -X POST $BASE/profiles -H 'Content-Type: application/json' \
   -d '{\"user_id\":\"test-999\",\"display_name\":\"Test User\",\"consent_behavioral\":true}'" \
  '"test-999"'

check "Get profile" "curl -sf $BASE/profiles/test-999" '"display_name"'

check "Update preferences" \
  "curl -sf -X PATCH $BASE/profiles/test-999/preferences -H 'Content-Type: application/json' \
   -d '{\"preferences\":{\"formality\":0.8}}'" \
  '"formality"'

check "Get persona" "curl -sf $BASE/profiles/exec-001/persona" '"persona"'

check "Chat with personalization" \
  "curl -sf -X POST $BASE/chat -H 'Content-Type: application/json' \
   -H 'X-User-Id: exec-001' \
   -d '{\"message\":\"What is Kubernetes?\",\"budget_fraction\":1.0}'" \
  '"persona"'

check "Chat - constrained budget" \
  "curl -sf -X POST $BASE/chat -H 'Content-Type: application/json' \
   -H 'X-User-Id: dev-002' \
   -d '{\"message\":\"What is Redis?\",\"budget_fraction\":0.2}'" \
  '"MINIMAL"'

check "Analytics stats" "curl -sf $BASE/analytics/stats" '"total_profiles"'

check "ADK context export" "curl -sf $BASE/profiles/exec-001/adk-context" '"user_id"'

check "Delete profile" "curl -sf -X DELETE $BASE/profiles/test-999" '"deleted"'

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo -e "\033[0;32mAll tests passed\033[0m" || echo -e "\033[0;31m$FAIL test(s) failed\033[0m"
SH
chmod +x scripts/test.sh

# =============================================================================
# STEP 9: Backend package init
# =============================================================================
cat > backend/__init__.py << 'PYEOF'
PYEOF
cat > backend/routers/__init__.py << 'PYEOF'
from .profiles import router as profiles_router
from .chat import router as chat_router
from .analytics import router as analytics_router
PYEOF
cat > backend/services/__init__.py << 'PYEOF'
from .inference import PreferenceInferenceEngine
from .router import AdaptiveAgentRouter
from .compressor import ContextCompressor
PYEOF
cat > backend/db/__init__.py << 'PYEOF'
from .store import UserProfileStore
PYEOF
cat > backend/models/__init__.py << 'PYEOF'
from .profile import UserProfile, PreferenceVector, PersonaType, PersonalizationContext
from .interaction import Interaction, InteractionFeedback
from .budget import BudgetState
PYEOF
cat > backend/middleware/__init__.py << 'PYEOF'
PYEOF
cat > backend/tasks/__init__.py << 'PYEOF'
PYEOF

# =============================================================================
# STEP 10: Install and verify
# =============================================================================
step "Installing Python dependencies"
cd "$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt
log "Python deps installed"

step "Installing frontend dependencies"
cd frontend
npm install --silent 2>/dev/null || npm install
cd ..
log "Frontend deps installed"

# =============================================================================
# STEP 11: Quick smoke test (no network)
# =============================================================================
step "Running Python import smoke test"
source .venv/bin/activate
python3 -c "
from backend.models.profile import PreferenceVector, PersonaType
from backend.services.router import AdaptiveAgentRouter
from backend.services.compressor import ContextCompressor
from backend.models.budget import BudgetState
import numpy as np

# Test persona routing
v = PreferenceVector(formality=0.9, verbosity=0.1, response_urgency=0.9)
router = AdaptiveAgentRouter()
persona = router.select_persona(v)
print(f'  Persona selection: {persona.value}')
assert persona == PersonaType.EXECUTIVE, f'Expected EXECUTIVE, got {persona.value}'

# Test budget-aware compression
compressor = ContextCompressor()
full_budget = BudgetState(max_tokens=8000, used_tokens=1000)  # fraction=0.875
ctx_rich = compressor.build_context(v, full_budget, persona)
assert ctx_rich.tier == 'RICH', f'Expected RICH, got {ctx_rich.tier}'

low_budget = BudgetState(max_tokens=8000, used_tokens=7500)  # fraction=0.0625
ctx_min = compressor.build_context(v, low_budget, persona)
assert ctx_min.tier == 'MINIMAL', f'Expected MINIMAL, got {ctx_min.tier}'

# Test differential privacy
noisy = v.add_laplace_noise(epsilon=0.1)
assert all(0.0 <= x <= 1.0 for x in noisy.to_list())

# Test ADK export format
up_dict = v.to_adk_metadata()
assert 'preference_vector_b64' in up_dict

print('  All smoke tests passed')
"

log "Smoke tests passed"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════╗"
echo "║         L54 Setup Complete                            ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Project:   $PROJECT_ROOT"
echo "║                                                       ║"
echo "║  Start:     cd l54-agent-personalization              ║"
echo "║             ./scripts/start.sh                        ║"
echo "║                                                       ║"
echo "║  Test:      ./scripts/test.sh                         ║"
echo "║                                                       ║"
echo "║  Docker:    cd docker && docker compose up --build    ║"
echo "║                                                       ║"
echo "║  Frontend → http://localhost:3054                     ║"
echo "║  Backend  → http://localhost:8054                     ║"
echo "║  API Docs → http://localhost:8054/docs                ║"
echo "╚═══════════════════════════════════════════════════════╝${NC}"