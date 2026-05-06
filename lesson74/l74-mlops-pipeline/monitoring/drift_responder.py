"""
DriftResponder — closed-loop controller.

Reads the rolling metric windows every `check_interval_seconds`.
When any threshold is breached it drives the DeploymentOrchestrator
through HEALTHY → DEGRADED → ROLLING_BACK → HEALTHY automatically.

Production note: replace _compute_drift_score() with a real KL-divergence
or PSI calculation against the baseline distribution stored in your
feature store (see L67).
"""
import asyncio
import statistics
import time
from collections import deque

from agent_service.config import settings
from agent_service.metrics import DRIFT_SCORE, ERROR_RATE
from monitoring.alert_router import AlertPayload, AlertWebhookRouter
from monitoring.orchestrator import DeploymentOrchestrator, DeploymentState


class DriftResponder:
    _BASELINE_LATENCY_MEAN = 0.40  # seconds — established in L68 drift monitoring

    def __init__(
        self,
        orchestrator: DeploymentOrchestrator,
        alert_router: AlertWebhookRouter,
    ) -> None:
        self.orchestrator = orchestrator
        self.alert_router = alert_router
        self._req_ts: deque[float] = deque(maxlen=200)
        self._err_ts: deque[float] = deque(maxlen=200)
        self._latencies: deque[float] = deque(maxlen=200)

    # ── called by agent_service/main.py after each request ────────────────
    def record_request(self, latency: float, success: bool, variant: str = "stable") -> None:
        now = time.time()
        self._req_ts.append(now)
        self._latencies.append(latency)
        if not success:
            self._err_ts.append(now)

    # ── metric computations ───────────────────────────────────────────────
    def _compute_error_rate(self) -> float:
        if not self._req_ts:
            return 0.0
        cutoff = time.time() - 60.0
        recent_req = sum(1 for t in self._req_ts if t > cutoff)
        recent_err = sum(1 for t in self._err_ts if t > cutoff)
        return recent_err / max(recent_req, 1)

    def _compute_p95_latency(self) -> float:
        if not self._latencies:
            return 0.0
        s = sorted(self._latencies)
        return s[int(len(s) * 0.95)]

    def _compute_drift_score(self) -> float:
        """
        Approximates drift as normalised deviation from baseline mean + variance penalty.
        Production: replace with KL-divergence against baseline distribution snapshot.
        """
        if len(self._latencies) < 5:
            return 0.0
        mean = statistics.mean(self._latencies)
        var = statistics.variance(self._latencies) if len(self._latencies) > 1 else 0.0
        raw = (
            abs(mean - self._BASELINE_LATENCY_MEAN) / (self._BASELINE_LATENCY_MEAN + 1e-9)
            + var * 0.5
        )
        return min(raw, 1.0)

    # ── main monitoring loop ───────────────────────────────────────────────
    async def run(self, shutdown_event: asyncio.Event) -> None:
        while not shutdown_event.is_set():
            await asyncio.sleep(settings.check_interval_seconds)
            if self.orchestrator.state not in (
                DeploymentState.HEALTHY,
                DeploymentState.DEGRADED,
            ):
                continue

            variant = settings.active_variant
            drift = self._compute_drift_score()
            err_rate = self._compute_error_rate()
            p95 = self._compute_p95_latency()

            DRIFT_SCORE.labels(variant=variant).set(drift)
            ERROR_RATE.labels(variant=variant).set(err_rate)

            should_degrade = (
                drift > settings.drift_threshold
                or err_rate > settings.error_rate_threshold
                or (p95 > settings.p95_latency_threshold and p95 > 0.5)
            )

            if should_degrade and self.orchestrator.state == DeploymentState.HEALTHY:
                reason = (
                    f"drift={drift:.3f} err_rate={err_rate:.3f} p95={p95:.3f}s"
                )
                await self.orchestrator.transition(DeploymentState.DEGRADED, reason)
                await self.alert_router.dispatch(
                    AlertPayload(
                        severity="HIGH",
                        trigger="threshold_breach",
                        value=drift,
                        threshold=settings.drift_threshold,
                        variant=variant,
                    )
                )
                await asyncio.sleep(5)
                await self.orchestrator.transition(
                    DeploymentState.ROLLING_BACK, "auto_rollback_triggered"
                )
                await asyncio.sleep(3)
                # Reset windows after rollback
                self._latencies.clear()
                self._err_ts.clear()
                await self.orchestrator.transition(
                    DeploymentState.HEALTHY, "rollback_complete"
                )
                await self.alert_router.dispatch(
                    AlertPayload(
                        severity="LOW",
                        trigger="rollback_complete",
                        value=0.0,
                        threshold=settings.drift_threshold,
                        variant=variant,
                    )
                )
