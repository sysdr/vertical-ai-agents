"""Database query helpers for L69 observability service."""
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "traces.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_traces(limit: int = 50, offset: int = 0) -> List[Dict]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM traces
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    return [dict(r) for r in rows]


def get_trace(trace_id: str) -> Optional[Dict]:
    with _connect() as conn:
        t = conn.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if not t:
            return None
        spans = conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,)
        ).fetchall()

    td = dict(t)
    span_list = []
    for s in spans:
        sd = dict(s)
        sd["attributes"] = json.loads(sd.get("attributes", "{}"))
        sd["events"]     = json.loads(sd.get("events", "[]"))
        span_list.append(sd)
    td["spans"] = span_list
    return td


def get_metrics() -> Dict[str, Any]:
    with _connect() as conn:
        agg = conn.execute("""
            SELECT
                COUNT(*)                                          AS total_requests,
                AVG(duration_ms)                                  AS avg_latency_ms,
                COALESCE(SUM(total_cost_usd), 0)                  AS total_cost_usd,
                COALESCE(SUM(input_tokens), 0)                    AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0)                   AS total_output_tokens,
                SUM(CASE WHEN status='ERROR' THEN 1 ELSE 0 END)   AS error_count
            FROM traces
            WHERE created_at > strftime('%s','now') - 3600
        """).fetchone()

        lats = [r["duration_ms"] for r in conn.execute("""
            SELECT duration_ms FROM spans
            WHERE name = 'agent.llm_call'
              AND created_at > strftime('%s','now') - 3600
            ORDER BY duration_ms
        """).fetchall()]

        cost_trend = conn.execute("""
            SELECT
                (created_at / 60) * 60  AS bucket,
                SUM(total_cost_usd)     AS cost,
                COUNT(*)                AS reqs
            FROM traces
            WHERE created_at > strftime('%s','now') - 600
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()

        recent_breaches = conn.execute("""
            SELECT metric, value, threshold, trace_id, created_at
            FROM metric_snapshots
            WHERE breached = 1
            ORDER BY created_at DESC
            LIMIT 10
        """).fetchall()

    def pct(lst: List[float], p: int) -> float:
        if not lst:
            return 0.0
        return round(lst[min(int(len(lst) * p / 100), len(lst) - 1)], 2)

    total = int(agg["total_requests"] or 0)
    errors = int(agg["error_count"] or 0)
    return {
        "total_requests":     total,
        "error_count":        errors,
        "error_rate_pct":     round(errors / total * 100, 2) if total else 0.0,
        "avg_latency_ms":     round(float(agg["avg_latency_ms"] or 0), 2),
        "p50_latency_ms":     pct(lats, 50),
        "p95_latency_ms":     pct(lats, 95),
        "p99_latency_ms":     pct(lats, 99),
        "total_cost_usd":     round(float(agg["total_cost_usd"] or 0), 6),
        "total_input_tokens": int(agg["total_input_tokens"] or 0),
        "total_output_tokens":int(agg["total_output_tokens"] or 0),
        "cost_trend":         [{"t": r["bucket"], "cost": round(r["cost"] or 0, 6), "reqs": r["reqs"]} for r in cost_trend],
        "recent_breaches":    [dict(r) for r in recent_breaches],
    }
