from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class FailureType(str, Enum):
    COMPLIANCE_VIOLATION = "compliance_violation"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    RETRIEVAL_ERROR = "retrieval_error"
    UNKNOWN = "unknown"

class FailureSignal(BaseModel):
    type: FailureType
    confidence: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    violated_rules: List[str] = Field(default_factory=list)

class ValidationResult(BaseModel):
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    failure_signal: Optional[FailureSignal] = None
    message: str = ""

class QueryPlan(BaseModel):
    original_query: str
    reformulated_query: Optional[str] = None
    strategy: str = "standard"
    sub_queries: List[str] = Field(default_factory=list)

class RetrievalContext(BaseModel):
    query: str
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    total_latency_ms: float = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetryMetrics(BaseModel):
    attempt_number: int
    delay_seconds: float
    reformulation_applied: bool
    validation_result: Optional[ValidationResult] = None
    timestamp: float
