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
