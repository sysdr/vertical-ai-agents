"""
DeploymentOrchestrator — drives the deployment FSM and persists every
state transition to SQLite in WAL mode.

Valid transitions:
  PENDING → DEPLOYING → HEALTHY ⇄ DEGRADED → ROLLING_BACK → HEALTHY
                                                           ↘ FAILED
"""
import asyncio
import time
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import aiosqlite

from agent_service.metrics import DEPLOYMENT_STATE


class DeploymentState(IntEnum):
    PENDING = 0
    DEPLOYING = 1
    HEALTHY = 2
    DEGRADED = 3
    ROLLING_BACK = 4
    FAILED = 5


_VALID: dict[DeploymentState, set[DeploymentState]] = {
    DeploymentState.PENDING:      {DeploymentState.DEPLOYING},
    DeploymentState.DEPLOYING:    {DeploymentState.HEALTHY, DeploymentState.FAILED},
    DeploymentState.HEALTHY:      {DeploymentState.DEGRADED},
    DeploymentState.DEGRADED:     {DeploymentState.ROLLING_BACK, DeploymentState.HEALTHY},
    DeploymentState.ROLLING_BACK: {DeploymentState.HEALTHY, DeploymentState.FAILED},
    DeploymentState.FAILED:       {DeploymentState.PENDING},
}


@dataclass
class StateEvent:
    from_state: DeploymentState
    to_state: DeploymentState
    reason: str
    timestamp: float = field(default_factory=time.time)


class DeploymentOrchestrator:
    def __init__(self, db_path: str = "data/deployment.db") -> None:
        self.db_path = db_path
        self.state = DeploymentState.PENDING
        self._db: Optional[aiosqlite.Connection] = None
        self._listeners: list[asyncio.Queue] = []

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS deployment_states (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                from_state TEXT NOT NULL,
                to_state   TEXT NOT NULL,
                reason     TEXT,
                ts         REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker (
                id                    INTEGER PRIMARY KEY CHECK (id = 1),
                is_open               INTEGER DEFAULT 0,
                rollback_timestamps   TEXT    DEFAULT '[]'
            )
        """)
        await self._db.execute(
            "INSERT OR IGNORE INTO circuit_breaker (id) VALUES (1)"
        )
        await self._db.commit()

    async def run(self) -> None:
        """Called once on startup — drives PENDING → DEPLOYING → HEALTHY."""
        await self._init_db()
        await self.transition(DeploymentState.DEPLOYING, "startup_sequence")
        await asyncio.sleep(2.0)
        await self.transition(DeploymentState.HEALTHY, "warmup_complete")

    async def drain(self) -> None:
        if self._db:
            await self._db.close()

    # ── FSM ────────────────────────────────────────────────────────────────
    async def transition(self, new_state: DeploymentState, reason: str = "") -> bool:
        if new_state not in _VALID.get(self.state, set()):
            return False
        event = StateEvent(self.state, new_state, reason)
        self.state = new_state
        DEPLOYMENT_STATE.set(int(new_state))
        await self._persist(event)
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return True

    async def _persist(self, event: StateEvent) -> None:
        if self._db:
            await self._db.execute(
                "INSERT INTO deployment_states (from_state, to_state, reason, ts) VALUES (?,?,?,?)",
                (event.from_state.name, event.to_state.name, event.reason, event.timestamp),
            )
            await self._db.commit()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._listeners.append(q)
        return q

    # ── queries ────────────────────────────────────────────────────────────
    async def get_history(self, limit: int = 50) -> list[dict]:
        if not self._db:
            return []
        async with self._db.execute(
            "SELECT from_state, to_state, reason, ts FROM deployment_states ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"from": r[0], "to": r[1], "reason": r[2], "ts": r[3]} for r in rows]
