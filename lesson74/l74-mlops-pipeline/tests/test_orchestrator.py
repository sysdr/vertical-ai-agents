"""Tests for DeploymentOrchestrator FSM."""
import asyncio
import pytest
from monitoring.orchestrator import DeploymentOrchestrator, DeploymentState


@pytest.fixture
async def orch(tmp_path):
    o = DeploymentOrchestrator(db_path=str(tmp_path / "test.db"))
    yield o
    await o.drain()


@pytest.mark.asyncio
async def test_valid_transition_sequence(orch):
    await orch._init_db()
    assert await orch.transition(DeploymentState.DEPLOYING, "test") is True
    assert orch.state == DeploymentState.DEPLOYING
    assert await orch.transition(DeploymentState.HEALTHY, "test") is True
    assert orch.state == DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_invalid_transition_blocked(orch):
    await orch._init_db()
    # PENDING cannot go directly to HEALTHY
    result = await orch.transition(DeploymentState.HEALTHY, "bad")
    assert result is False
    assert orch.state == DeploymentState.PENDING


@pytest.mark.asyncio
async def test_degraded_to_rolling_back(orch):
    await orch._init_db()
    await orch.transition(DeploymentState.DEPLOYING, "x")
    await orch.transition(DeploymentState.HEALTHY, "x")
    await orch.transition(DeploymentState.DEGRADED, "drift")
    assert await orch.transition(DeploymentState.ROLLING_BACK, "rollback") is True


@pytest.mark.asyncio
async def test_history_persisted(orch):
    await orch._init_db()
    await orch.transition(DeploymentState.DEPLOYING, "start")
    history = await orch.get_history()
    assert len(history) == 1
    assert history[0]["to"] == "DEPLOYING"


@pytest.mark.asyncio
async def test_subscriber_receives_events(orch):
    await orch._init_db()
    q = orch.subscribe()
    await orch.transition(DeploymentState.DEPLOYING, "startup")
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event.to_state == DeploymentState.DEPLOYING
