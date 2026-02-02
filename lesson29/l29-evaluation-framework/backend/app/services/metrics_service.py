import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.models.schemas import MetricsResponse, RegressionResult

logger = logging.getLogger(__name__)

class MetricsService:
    """Aggregate and analyze metrics - stores latest evaluation results"""

    def __init__(self):
        self.metrics_history = []
        self._last_evaluation: Optional[Dict[str, Any]] = None
        self.baseline_metrics = {
            'tool_call_accuracy': 0.95,
            'rag_groundedness': 0.90,
            'p99_latency_ms': 500.0
        }

    async def initialize(self):
        logger.info("Metrics service initialized")

    def record_evaluation(self, result: Dict[str, Any]):
        """Store evaluation result so dashboard shows real data"""
        self._last_evaluation = result
        self.metrics_history.append({
            'timestamp': datetime.utcnow(),
            **{k: v for k, v in result.items() if k != 'latency_summary'}
        })

    async def get_summary(self) -> MetricsResponse:
        """Return metrics from last evaluation, or sensible defaults if none yet"""
        if self._last_evaluation:
            r = self._last_evaluation
            lat = r.get('latency_summary') or []
            p99 = max((l.get('p99_ms', 0) for l in lat if isinstance(l, dict)), default=0)
            avg = sum((l.get('mean_ms', 0) for l in lat if isinstance(l, dict))) / max(len(lat), 1)
            return MetricsResponse(
                overall_accuracy=(r.get('tool_call_accuracy', 0) + r.get('rag_groundedness', 0)) / 2,
                tool_call_accuracy=r.get('tool_call_accuracy', 0),
                rag_groundedness=r.get('rag_groundedness', 0),
                avg_latency_ms=avg if avg > 0 else 100,
                p99_latency_ms=p99 if p99 > 0 else 200,
                total_tests_run=r.get('total_tests', 0),
                last_updated=datetime.utcnow()
            )
        return MetricsResponse(
            overall_accuracy=0,
            tool_call_accuracy=0,
            rag_groundedness=0,
            avg_latency_ms=0,
            p99_latency_ms=0,
            total_tests_run=0,
            last_updated=datetime.utcnow()
        )

    async def get_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.get('timestamp', datetime.min) > cutoff]

    async def detect_regressions(self) -> List[RegressionResult]:
        current = await self.get_summary()
        results = []
        threshold = 0.05
        for name, base in [
            ('tool_call_accuracy', current.tool_call_accuracy),
            ('rag_groundedness', current.rag_groundedness),
            ('p99_latency_ms', current.p99_latency_ms)
        ]:
            bl = self.baseline_metrics.get(name, 0) or 0.01
            change = ((base - bl) / bl) * 100 if bl else 0
            is_reg = (change < -threshold * 100) if name != 'p99_latency_ms' else (change > threshold * 100)
            results.append(RegressionResult(
                metric_name=name,
                current_value=base,
                baseline_value=bl,
                change_percent=change,
                is_regression=is_reg,
                threshold=threshold * 100
            ))
        return results
