"""
AlertWebhookRouter — dispatches structured JSON alerts to a configured
webhook URL with exponential-backoff retry.  Stores in-memory for the
dashboard; swap for persistent queue in production.
"""
import asyncio
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class AlertPayload:
    severity: str        # LOW | MEDIUM | HIGH | CRITICAL
    trigger: str         # e.g. "threshold_breach"
    value: float
    threshold: float
    variant: str = "stable"
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"{self.trigger}: value={self.value:.3f} "
                f"> threshold={self.threshold:.3f} variant={self.variant}"
            )


class AlertWebhookRouter:
    def __init__(self, webhook_url: str = "", max_retries: int = 3) -> None:
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self._log: list[AlertPayload] = []

    async def dispatch(self, payload: AlertPayload) -> bool:
        self._log.append(payload)
        print(f"[ALERT:{payload.severity}] {payload.message}")
        if not self.webhook_url:
            return True
        body = {
            "severity": payload.severity,
            "trigger": payload.trigger,
            "value": payload.value,
            "threshold": payload.threshold,
            "variant": payload.variant,
            "message": payload.message,
            "timestamp": payload.timestamp,
        }
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.post(self.webhook_url, json=body)
                    if r.status_code < 300:
                        return True
            except Exception as exc:
                wait = 2 ** attempt
                print(f"[ALERT] attempt {attempt + 1} failed: {exc}; retry in {wait}s")
                await asyncio.sleep(wait)
        return False

    def get_recent(self, n: int = 20) -> list[dict]:
        return [
            {
                "severity": a.severity,
                "trigger": a.trigger,
                "value": a.value,
                "threshold": a.threshold,
                "message": a.message,
                "ts": a.timestamp,
            }
            for a in self._log[-n:]
        ]
