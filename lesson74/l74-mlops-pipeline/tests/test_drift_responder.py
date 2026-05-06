"""Tests for DriftResponder metrics computation."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from monitoring.drift_responder import DriftResponder
from monitoring.orchestrator import DeploymentOrchestrator, DeploymentState


def make_responder(tmp_path):
    orch = DeploymentOrchestrator(db_path=str(tmp_path / "dr.db"))
    orch.state = DeploymentState.HEALTHY
    router = MagicMock()
    router.dispatch = AsyncMock()
    return DriftResponder(orch, router), orch, router


def test_no_drift_with_baseline_latencies(tmp_path):
    resp, _, _ = make_responder(tmp_path)
    for _ in range(20):
        resp.record_request(latency=0.40, success=True)
    score = resp._compute_drift_score()
    assert score < 0.35, f"Expected low drift, got {score}"


def test_high_drift_with_spiked_latencies(tmp_path):
    resp, _, _ = make_responder(tmp_path)
    for _ in range(20):
        resp.record_request(latency=4.0, success=True)
    score = resp._compute_drift_score()
    assert score > 0.35, f"Expected high drift, got {score}"


def test_error_rate_computation(tmp_path):
    resp, _, _ = make_responder(tmp_path)
    for _ in range(10):
        resp.record_request(latency=0.4, success=True)
    for _ in range(5):
        resp.record_request(latency=0.4, success=False)
    rate = resp._compute_error_rate()
    assert abs(rate - 5 / 15) < 0.02


def test_p95_latency_computation(tmp_path):
    resp, _, _ = make_responder(tmp_path)
    for i in range(100):
        resp.record_request(latency=float(i) / 100, success=True)
    p95 = resp._compute_p95_latency()
    assert 0.90 <= p95 <= 1.0
