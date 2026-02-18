"""
TraceQueryEngine: reconstruct waterfall timelines, risk timelines, and span search.
"""
from __future__ import annotations
import json, os, time
from typing import List, Optional
import aiosqlite

DB_PATH = os.getenv("TRACE_DB_PATH", "data/traces/audit.db")


class TraceQueryEngine:
    async def get_traces(self, limit: int = 50) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def get_waterfall(self, trace_id: str) -> dict:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ) as c:
                trace = dict(await c.fetchone() or {})
            async with db.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC",
                (trace_id,),
            ) as c:
                spans = [dict(r) for r in await c.fetchall()]
        for s in spans:
            s["input_summary"] = json.loads(s.get("input_summary") or "{}")
            s["output_summary"] = json.loads(s.get("output_summary") or "{}")
            s["compliance_flags"] = json.loads(s.get("compliance_flags") or "[]")
        trace["spans"] = spans
        return trace

    async def get_risk_timeline(self, hours: int = 24) -> List[dict]:
        since = time.monotonic() - (hours * 3600)
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT agent_name, AVG(risk_score) as avg_risk, MAX(risk_score) as max_risk,
                          COUNT(*) as count
                   FROM spans
                   WHERE started_at > ?
                   GROUP BY agent_name
                   ORDER BY avg_risk DESC""",
                (since,),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def search_by_decision(self, decision_fragment: str, limit: int = 20) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM spans WHERE decision LIKE ? ORDER BY started_at DESC LIMIT ?",
                (f"%{decision_fragment}%", limit),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def get_high_risk_traces(self, threshold: float = 0.7, limit: int = 20) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT DISTINCT t.* FROM traces t
                   JOIN spans s ON t.trace_id = s.trace_id
                   WHERE s.risk_score >= ?
                   ORDER BY t.created_at DESC LIMIT ?""",
                (threshold, limit),
            ) as c:
                return [dict(r) for r in await c.fetchall()]

    async def get_agent_latency_stats(self) -> List[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT agent_name,
                          ROUND(AVG(latency_ms),2) as avg_ms,
                          ROUND(MIN(latency_ms),2) as min_ms,
                          ROUND(MAX(latency_ms),2) as max_ms,
                          COUNT(*) as total_spans
                   FROM spans GROUP BY agent_name""",
            ) as c:
                return [dict(r) for r in await c.fetchall()]
