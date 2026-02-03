from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class PipelineStage(str, Enum):
    RECEIVED = "received"
    EMBEDDING = "embedding"
    RETRIEVING = "retrieving"
    AUGMENTING = "augmenting"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

class TraceSpan(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    stage: PipelineStage
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class MetricPoint(BaseModel):
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    trace_id: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    trace_id: str
    stage: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceStats(BaseModel):
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0

class RAGMetrics(BaseModel):
    embedding_latency_ms: float
    retrieval_latency_ms: float
    context_tokens: int
    generation_latency_ms: float
    completion_tokens: int
    total_latency_ms: float
    cost_usd: float
    chunks_retrieved: int
    avg_similarity_score: float
