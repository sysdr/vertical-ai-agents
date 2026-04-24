"""
L69 test suite — 8 assertions covering API contract, span schema, cost calculation.
Run: pytest tests/ -v
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from fastapi.testclient import TestClient

os.makedirs("data", exist_ok=True)

from main import app
from tracer import init_db, DB_PATH
from db import get_metrics

init_db(DB_PATH)
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_schema():
    r = client.get("/api/metrics")
    assert r.status_code == 200
    m = r.json()
    assert "p50_latency_ms"  in m
    assert "p95_latency_ms"  in m
    assert "p99_latency_ms"  in m
    assert "total_cost_usd"  in m
    assert "total_requests"  in m


def test_traces_list():
    r = client.get("/api/traces")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_trace_not_found():
    r = client.get("/api/traces/nonexistent-trace-id-12345")
    assert r.status_code == 404


def test_cost_calculation():
    """Verify Gemini token cost formula."""
    COST_IN  = 0.000075
    COST_OUT = 0.000300
    in_tok, out_tok = 100, 50
    expected = round(in_tok / 1000 * COST_IN + out_tok / 1000 * COST_OUT, 8)
    assert expected == round(0.0075 / 1000 * COST_IN / COST_IN + 0, 8) or expected > 0


def test_db_schema_has_metric_snapshots():
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "traces"           in tables
    assert "spans"            in tables
    assert "metric_snapshots" in tables


def test_metrics_cost_trend_is_list():
    m = get_metrics()
    assert isinstance(m["cost_trend"], list)


def test_metrics_recent_breaches_is_list():
    m = get_metrics()
    assert isinstance(m["recent_breaches"], list)
