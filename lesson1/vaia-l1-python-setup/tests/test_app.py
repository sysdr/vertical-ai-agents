"""
Application and dashboard tests
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

client = TestClient(app)

def test_root_endpoint():
    """Test dashboard HTML is served"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VAIA L1 Dashboard" in response.text

def test_metrics_endpoint():
    """Test metrics API endpoint"""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_total" in data
    assert "demo_executions" in data
    assert "successful_operations" in data
    assert "uptime_seconds" in data
    assert isinstance(data["requests_total"], int)
    assert isinstance(data["demo_executions"], int)

def test_demo_endpoint():
    """Test demo endpoint updates metrics"""
    # Get initial metrics
    initial_response = client.get("/api/metrics")
    initial_data = initial_response.json()
    initial_demo_count = initial_data["demo_executions"]
    
    # Run demo
    demo_response = client.post("/api/demo", json={"action": "run"})
    assert demo_response.status_code == 200
    demo_data = demo_response.json()
    assert demo_data["status"] == "success"
    
    # Verify metrics updated
    updated_response = client.get("/api/metrics")
    updated_data = updated_response.json()
    assert updated_data["demo_executions"] == initial_demo_count + 1
    assert updated_data["requests_total"] > initial_data["requests_total"]

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "metrics" in data

def test_metrics_not_zero_after_demo():
    """Test that metrics are not zero after running demo"""
    # Run a few demos
    for _ in range(5):
        client.post("/api/demo", json={"action": "run"})
    
    response = client.get("/api/metrics")
    data = response.json()
    assert data["demo_executions"] > 0, "Demo executions should not be zero"
    assert data["requests_total"] > 0, "Total requests should not be zero"
    assert data["successful_operations"] > 0, "Successful operations should not be zero"
