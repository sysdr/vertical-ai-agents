from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum

class PipelineStage(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RETRIEVING = "RETRIEVING"
    VALIDATING = "VALIDATING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class StageResult:
    success: bool
    data: Optional[Dict] = None
    tokens_used: int = 0
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class PipelineState:
    query: str
    stage: PipelineStage = PipelineStage.IDLE
    plan: Optional[Dict] = None
    documents: List[Dict] = field(default_factory=list)
    validation_scores: Optional[Dict] = None
    response: Optional[str] = None
    total_tokens: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "stage": self.stage.value,
            "plan": self.plan,
            "documents": self.documents,
            "validation_scores": self.validation_scores,
            "response": self.response,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
            "metadata": self.metadata
        }
