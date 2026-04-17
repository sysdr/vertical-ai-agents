"""
L64 – Container health endpoint tests.
These run in the CI job before docker build is triggered.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from src.main import app


@pytest.mark.asyncio
async def test_liveness_returns_ok():
    """Liveness must always return 200 – never expensive checks."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_503_when_agent_not_ready():
    """Before warm-up completes, readiness must return 503."""
    import src.main as main_module
    original = main_module.agent
    main_module.agent = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/health/ready")
        assert r.status_code == 503
    finally:
        main_module.agent = original


@pytest.mark.asyncio
async def test_metrics_endpoint_reachable():
    """Prometheus /metrics endpoint must be reachable."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/metrics", follow_redirects=True)
    assert r.status_code == 200
    assert b"vaia_requests_total" in r.content or b"# HELP" in r.content


@pytest.mark.asyncio
async def test_infer_503_when_not_ready():
    """Inference must return 503 while agent is not ready."""
    import src.main as main_module
    original = main_module.agent
    main_module.agent = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/v1/agent/infer", json={"query": "hello"})
        assert r.status_code == 503
    finally:
        main_module.agent = original
