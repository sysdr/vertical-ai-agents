"""
L44 Pydantic models — evaluation data structures.
Connects directly to L43 trace format.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class EvalState(str, Enum):
    IDLE = "idle"
    DATASET_LOADING = "dataset_loading"
    RAG_EXECUTING = "rag_executing"
    TRACES_COLLECTED = "traces_collected"
    EVALUATING = "evaluating"
    GATE_CHECK = "gate_check"
    REPORT_GENERATED = "report_generated"
    DEPLOYED = "deployed"
    BLOCKED = "blocked"
    ERROR = "error"


class MetricScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None

    def all_above_threshold(self, thresholds: dict) -> tuple[bool, dict]:
        failures = {}
        scores = self.model_dump()
        for metric, threshold in thresholds.items():
            score = scores.get(metric)
            if score is not None and score < threshold:
                failures[metric] = {"score": score, "threshold": threshold}
        return len(failures) == 0, failures


class TraceRecord(BaseModel):
    """L43 pipeline trace — consumed directly by L44 evaluator"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    retrieved_contexts: list[str]
    answer: str
    ground_truth: Optional[str] = None
    validation_status: Optional[str] = None
    correction_count: int = 0
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvalItem(BaseModel):
    trace: TraceRecord
    scores: Optional[MetricScores] = None
    evaluation_reasoning: Optional[dict] = None


class EvaluationReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_name: str
    state: EvalState = EvalState.IDLE
    items: list[EvalItem] = []
    aggregate: Optional[MetricScores] = None
    gate_passed: Optional[bool] = None
    blocking_metrics: dict = {}
    total_questions: int = 0
    evaluated_questions: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class EvaluationRequest(BaseModel):
    run_name: str = Field(default="eval-run")
    num_test_questions: int = Field(default=10, ge=1, le=50)
    corpus_source: str = Field(default="l43_chromadb")
    use_synthetic_dataset: bool = Field(default=True)
    custom_questions: Optional[list[str]] = None
    custom_ground_truths: Optional[list[str]] = None


DEPLOYMENT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
    "context_precision": 0.70,
}
