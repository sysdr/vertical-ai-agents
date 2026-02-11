from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Document(BaseModel):
    id: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    metadata: dict = Field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
class ValidationScore(BaseModel):
    consistency_score: float = Field(ge=0.0, le=1.0)
    contradictions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    pair_id: str
    cached: bool = False
    
class ValidatedDocument(Document):
    validation_status: str  # "passed", "warning", "failed", "skipped"
    consistency_score: float = Field(ge=0.0, le=1.0)
    contradiction_flags: List[str] = Field(default_factory=list)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class ValidationReport(BaseModel):
    overall_consistency: float = Field(ge=0.0, le=1.0)
    total_pairs: int
    validated_pairs: int
    high_conflict_pairs: List[dict] = Field(default_factory=list)
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int
    documents: List[ValidatedDocument]
