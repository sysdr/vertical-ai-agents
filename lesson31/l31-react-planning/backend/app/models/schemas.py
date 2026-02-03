from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Reasoning(BaseModel):
    thought: str = Field(..., description="LLM reasoning about current state")
    action: str = Field(..., description="Proposed next action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    is_terminal: bool = Field(default=False, description="Whether this completes the task")

class Observation(BaseModel):
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    task: Optional[str] = None
    previous_thought: Optional[str] = None
    previous_action: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

class PlanningStep(BaseModel):
    iteration: int
    observation: Observation
    reasoning: Reasoning
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PlanningTrace(BaseModel):
    task_id: str
    task: str
    steps: List[PlanningStep]
    status: str  # RUNNING, COMPLETED, FAILED
    total_iterations: int
    total_tokens: int
    total_latency_ms: float
    started_at: datetime
    completed_at: Optional[datetime] = None

class PlanningRequest(BaseModel):
    task: str
    max_iterations: int = 10
    initial_context: Dict[str, Any] = Field(default_factory=dict)

class PlanningResponse(BaseModel):
    task_id: str
    trace: PlanningTrace
