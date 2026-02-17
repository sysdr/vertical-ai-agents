from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class SynthesisState(str, Enum):
    AWAITING_CONTEXT = "awaiting_context"
    PLANNING_SYNTHESIS = "planning_synthesis"
    GENERATING_CLAIMS = "generating_claims"
    CLAIMS_COMPLETE = "claims_complete"
    MAPPING_CITATIONS = "mapping_citations"
    ASSEMBLING_RESPONSE = "assembling_response"
    VALIDATING_OUTPUT = "validating_output"
    READY_TO_SEND = "ready_to_send"
    REGENERATE_REQUIRED = "regenerate_required"

class Chunk(BaseModel):
    id: str
    content: str
    source: str
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0

class Citation(BaseModel):
    chunk_id: str
    source: str
    section: Optional[str] = None
    text_excerpt: str
    confidence: float

class ReasoningStep(BaseModel):
    step_name: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)

class SynthesisRequest(BaseModel):
    query: str
    validated_chunks: List[Chunk]
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_context: Dict[str, Any] = Field(default_factory=dict)

class SynthesisResponse(BaseModel):
    response_text: str
    citations: List[Citation]
    reasoning_chain: List[ReasoningStep]
    overall_confidence: float
    synthesis_time_ms: float
    state: SynthesisState
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SynthesisEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    step_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    latency_ms: float
    trace_id: str
