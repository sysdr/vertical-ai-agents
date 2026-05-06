"""Tests for AlertWebhookRouter."""
import pytest
from monitoring.alert_router import AlertPayload, AlertWebhookRouter


@pytest.mark.asyncio
async def test_dispatch_no_webhook_always_succeeds():
    router = AlertWebhookRouter(webhook_url="")
    payload = AlertPayload(
        severity="HIGH",
        trigger="test",
        value=0.5,
        threshold=0.35,
    )
    result = await router.dispatch(payload)
    assert result is True


def test_get_recent_returns_last_n():
    router = AlertWebhookRouter()
    import asyncio
    for i in range(25):
        payload = AlertPayload(severity="LOW", trigger=f"t{i}", value=float(i), threshold=0.35)
        router._log.append(payload)
    recent = router.get_recent(10)
    assert len(recent) == 10
    assert recent[-1]["trigger"] == "t24"
