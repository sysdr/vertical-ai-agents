import asyncio
import time
import statistics
from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import aiosqlite
from models.telemetry import MetricPoint, PerformanceStats

class MetricsCollector:
    """Production-grade metrics aggregation with time-series storage"""
    
    def __init__(self, db_path: str = "data/metrics.db", flush_interval: int = 10):
        self.db_path = db_path
        self.flush_interval = flush_interval
        self.metrics_buffer: List[MetricPoint] = []
        self.latency_samples: deque = deque(maxlen=10000)
        self.active_traces: Dict[str, float] = {}
        self.flush_task: Optional[asyncio.Task] = None
        
        # In-memory aggregates for fast queries
        self.counters = defaultdict(float)
        self.histograms = defaultdict(list)
        
    async def initialize(self):
        """Create metrics database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    trace_id TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON metrics(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_name 
                ON metrics(metric_name)
            """)
            await db.commit()
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self._flush_loop())
    
    def record(self, metric_name: str, value: float, labels: Dict[str, str] = None, trace_id: str = None):
        """Record a metric point"""
        point = MetricPoint(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            trace_id=trace_id
        )
        self.metrics_buffer.append(point)
        
        # Update in-memory aggregates
        if metric_name.endswith("_latency_ms"):
            self.latency_samples.append(value)
            self.histograms[metric_name].append(value)
        else:
            self.counters[metric_name] += value
    
    def increment(self, metric_name: str, labels: Dict[str, str] = None, amount: float = 1.0):
        """Increment a counter metric"""
        self.record(metric_name, amount, labels)
    
    async def start_trace(self, trace_id: str):
        """Mark trace start time"""
        self.active_traces[trace_id] = time.perf_counter()
    
    async def end_trace(self, trace_id: str) -> float:
        """Calculate and record trace duration"""
        if trace_id in self.active_traces:
            duration_ms = (time.perf_counter() - self.active_traces[trace_id]) * 1000
            self.record("total_latency_ms", duration_ms, trace_id=trace_id)
            del self.active_traces[trace_id]
            return duration_ms
        return 0.0
    
    async def get_stats(self, time_window_minutes: int = 60) -> PerformanceStats:
        """Calculate aggregate statistics"""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        
        latencies = [v for v in self.latency_samples if v > 0]
        
        if not latencies:
            return PerformanceStats()
        
        sorted_latencies = sorted(latencies)
        
        return PerformanceStats(
            total_requests=len(latencies),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
            p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
            p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)],
            total_tokens=int(self.counters.get("completion_tokens", 0)),
            total_cost_usd=self.counters.get("llm_cost_usd", 0.0),
            error_rate=self.counters.get("errors", 0) / max(len(latencies), 1),
            cache_hit_rate=self.counters.get("cache_hits", 0) / max(len(latencies), 1)
        )
    
    async def get_timeseries(self, metric_name: str, minutes: int = 60) -> List[Dict]:
        """Retrieve time-series data for charting"""
        cutoff = (datetime.now() - timedelta(minutes=minutes)).timestamp()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT timestamp, value FROM metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp
            """, (metric_name, cutoff)) as cursor:
                rows = await cursor.fetchall()
                return [{"timestamp": r[0], "value": r[1]} for r in rows]
    
    async def _flush_loop(self):
        """Background task to flush metrics to database"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush()
    
    async def _flush(self):
        """Write buffered metrics to database"""
        if not self.metrics_buffer:
            return
        
        buffer_copy = self.metrics_buffer[:]
        self.metrics_buffer.clear()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany("""
                INSERT INTO metrics (timestamp, metric_name, value, labels, trace_id)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (
                    m.timestamp.timestamp(),
                    m.metric_name,
                    m.value,
                    str(m.labels),
                    m.trace_id
                ) for m in buffer_copy
            ])
            await db.commit()
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.flush_task:
            self.flush_task.cancel()
        await self._flush()

# Global metrics instance
metrics = MetricsCollector()
