from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TestCaseType(str, Enum):
    TOOL_CALL = "tool_call"
    RAG_GROUNDEDNESS = "rag_groundedness"
    LATENCY = "latency"

class TestCase(BaseModel):
    id: str
    type: TestCaseType
    query: str
    expected_tool: Optional[str] = None
    expected_params: Optional[Dict[str, Any]] = None
    expected_response: Optional[str] = None
    ground_truth_docs: Optional[List[str]] = None

class EvaluationRequest(BaseModel):
    test_suite_name: str
    iterations: int = Field(default=3, ge=1, le=1000)
    config: Optional[Dict[str, Any]] = None

class TestSuiteRequest(BaseModel):
    test_cases: List[TestCase]

class ToolCallResult(BaseModel):
    test_case_id: str
    actual_tool: Optional[str]
    expected_tool: str
    parameters_match: float
    success: bool
    latency_ms: float

class RAGGroundednessResult(BaseModel):
    test_case_id: str
    response: str
    groundedness_score: float
    unsupported_claims: List[str]
    retrieved_chunks: int
    success: bool

class LatencyResult(BaseModel):
    component: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    samples: int

class EvaluationResponse(BaseModel):
    test_suite: str
    total_tests: int
    passed: int
    failed: int
    tool_call_accuracy: float
    rag_groundedness: float
    latency_summary: List[LatencyResult]
    timestamp: datetime
    duration_seconds: float

class MetricsResponse(BaseModel):
    overall_accuracy: float
    tool_call_accuracy: float
    rag_groundedness: float
    avg_latency_ms: float
    p99_latency_ms: float
    total_tests_run: int
    last_updated: datetime

class RegressionResult(BaseModel):
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    is_regression: bool
    threshold: float
