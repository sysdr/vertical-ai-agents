"""
MASTraceLogger: async-safe, non-blocking trace emission.
Uses asyncio.Queue with WAL-mode SQLite persistence.
"""
from __future__ import annotations
import asyncio
import json
import time
import os
from pathlib import Path
from typing import Optional, Callable

import aiosqlite
from loguru import logger

from models.trace import TraceRecord

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "mas_traces_{time}.jsonl",
    format="{message}",
    serialize=True,
    rotation="100 MB",
    compression="gz",
    level="DEBUG",
    filter=lambda r: "trace_id" in r["extra"],
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    span_id     TEXT NOT NULL UNIQUE,
    parent_id   TEXT,
    agent_name  TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    duration_ns INTEGER NOT NULL DEFAULT 0,
    payload     TEXT NOT NULL DEFAULT '{}',
    prompt_tok  INTEGER NOT NULL DEFAULT 0,
    compl_tok   INTEGER NOT NULL DEFAULT 0,
    modality    TEXT NOT NULL DEFAULT 'text',
    error_msg   TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace   ON traces (trace_id);
CREATE INDEX IF NOT EXISTS idx_parent  ON traces (parent_id);
CREATE INDEX IF NOT EXISTS idx_agent   ON traces (agent_name);
"""

_DROPPED = 0


class MASTraceLogger:
    def __init__(self, db_path: str = "data/traces.db", max_queue: int = 10_000):
        self._db_path = db_path
        self._queue: asyncio.Queue[TraceRecord] = asyncio.Queue(maxsize=max_queue)
        self._writer: Optional[asyncio.Task] = None
        self._ws_callbacks: list[Callable] = []
        self._running = False

    def register_ws_callback(self, cb: Callable) -> None:
        self._ws_callbacks.append(cb)

    async def start(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        await self._init_db()
        self._running = True
        self._writer = asyncio.create_task(self._drain_loop(), name="trace-writer")
        logger.info("MASTraceLogger started", db=self._db_path)

    async def stop(self) -> None:
        self._running = False
        if self._writer:
            # flush remaining items
            await asyncio.sleep(0.1)
            self._writer.cancel()
            try:
                await self._writer
            except asyncio.CancelledError:
                pass
        logger.info("MASTraceLogger stopped")

    async def emit(self, record: TraceRecord) -> None:
        global _DROPPED
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            _DROPPED += 1
            logger.bind(trace_id=record.trace_id).warning(
                "Trace queue full — dropping record",
                span_id=record.span_id,
                dropped_total=_DROPPED,
            )

    async def _init_db(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            await db.executescript(_SCHEMA)
            await db.commit()

    async def _drain_loop(self) -> None:
        batch: list[TraceRecord] = []
        while self._running or not self._queue.empty():
            try:
                record = await asyncio.wait_for(self._queue.get(), timeout=0.05)
                batch.append(record)
                # drain eagerly up to 50 records
                while not self._queue.empty() and len(batch) < 50:
                    batch.append(self._queue.get_nowait())
            except asyncio.TimeoutError:
                pass

            if batch:
                await self._write_batch(batch)
                await self._broadcast_batch(batch)
                batch.clear()

    async def _write_batch(self, records: list[TraceRecord]) -> None:
        rows = [
            (
                r.trace_id, r.span_id, r.parent_id, r.agent_name, r.event_type,
                r.timestamp, r.duration_ns,
                json.dumps(r.payload), r.token_cost.prompt_tokens,
                r.token_cost.completion_tokens, r.modality, r.error_msg,
            )
            for r in records
        ]
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(
                """INSERT OR IGNORE INTO traces
                   (trace_id,span_id,parent_id,agent_name,event_type,timestamp,
                    duration_ns,payload,prompt_tok,compl_tok,modality,error_msg)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            await db.commit()

    async def _broadcast_batch(self, records: list[TraceRecord]) -> None:
        for cb in self._ws_callbacks:
            for r in records:
                try:
                    await cb(r.model_dump_json())
                except Exception:
                    pass

    async def fetch_traces(self, trace_id: str) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces WHERE trace_id=? ORDER BY timestamp ASC",
                (trace_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def list_traces(self, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT trace_id, MIN(timestamp) as started, COUNT(*) as spans,
                          SUM(duration_ns) as total_ns
                   FROM traces GROUP BY trace_id
                   ORDER BY started DESC LIMIT ?""",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    @property
    def dropped(self) -> int:
        return _DROPPED
