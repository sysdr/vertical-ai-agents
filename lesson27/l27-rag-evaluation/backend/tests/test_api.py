"""Basic API tests for L27 RAG Evaluation"""
import pytest
from fastapi.testclient import TestClient

# Import app - use lazy evaluator so no API key needed at import
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["lesson"] == "L27"
    assert data["status"] == "operational"


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_summary():
    """Test metrics summary returns non-zero baseline"""
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["avg_faithfulness"] > 0
    assert data["avg_relevance"] > 0
    assert data["avg_precision"] > 0
    assert data["avg_recall"] > 0


def test_benchmarks_sample_queries():
    """Test sample queries endpoint"""
    response = client.get("/api/benchmarks/sample-queries")
    assert response.status_code == 200
    data = response.json()
    assert "queries" in data
    assert len(data["queries"]) > 0


def test_evaluation_runs_empty():
    """Test list runs when empty"""
    response = client.get("/api/evaluation/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
