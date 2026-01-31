from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"

class EvaluationRequest(BaseModel):
    run_name: str
    queries: List[str]
    model_config: Optional[dict] = {}

class QueryResult(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevance: float
    context_precision: Optional[float]
    context_recall: Optional[float]

class EvaluationResponse(BaseModel):
    run_id: int
    run_name: str
    status: EvaluationStatus
    faithfulness_score: Optional[float]
    answer_relevance_score: Optional[float]
    context_precision_score: Optional[float]
    context_recall_score: Optional[float]
    total_queries: int
    completed_queries: int
    created_at: Optional[datetime] = None

class MetricsSummary(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    avg_precision: float
    avg_recall: float
    total_evaluations: int
    period: str
