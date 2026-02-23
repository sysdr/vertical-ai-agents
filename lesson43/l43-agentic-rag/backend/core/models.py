"""
L43: Core Pydantic models for Agentic RAG pipeline.
Every agent exchanges typed objects — no raw dicts.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


class PipelineState(str, Enum):
    IDLE        = "idle"
    PLANNING    = "planning"
    RETRIEVING  = "retrieving"
    VALIDATING  = "validating"
    SYNTHESIZING = "synthesizing"
    TRACING     = "tracing"
    DONE        = "done"
    ERROR       = "error"


class TraceEvent(BaseModel):
    trace_id:   str
    agent:      str
    state:      PipelineState
    input:      dict[str, Any]
    output:     dict[str, Any]
    latency_ms: float
    timestamp:  float = Field(default_factory=time.time)
    attempt:    int   = 1
    error:      Optional[str] = None


class SubQuery(BaseModel):
    text:     str
    strategy: str  # "semantic" | "keyword" | "hybrid"
    priority: int  # 1=high, 2=med, 3=low


class PlanResult(BaseModel):
    original_query: str
    sub_queries:    list[SubQuery]
    retrieval_mode: str  # "sequential" | "parallel"
    reasoning:      str


class Chunk(BaseModel):
    chunk_id:   str
    text:       str
    source:     str
    score:      float
    metadata:   dict[str, Any] = {}


class RetrievalResult(BaseModel):
    query:       str
    chunks:      list[Chunk]
    total_found: int
    latency_ms:  float


class ScoredChunk(BaseModel):
    chunk:         Chunk
    quality_score: float   # 0.0–1.0 relevance to original query
    risk_score:    float   # 0.0–1.0 (high = risky)
    passed:        bool


class ValidationResult(BaseModel):
    validated_chunks: list[ScoredChunk]
    avg_quality_score: float
    attempts:          int
    blocked_reason:    Optional[str] = None
    passed:            bool


class Citation(BaseModel):
    marker:   str   # e.g. "[1]"
    chunk_id: str
    source:   str
    text_snippet: str


class SynthesisResult(BaseModel):
    trace_id:     str
    query:        str
    response:     str
    citations:    list[Citation]
    tokens_used:  int
    latency_ms:   float
    quality_score: float
    pipeline_state: PipelineState


class QueryRequest(BaseModel):
    query:       str  = Field(..., min_length=3, max_length=1000)
    collection:  str  = "enterprise_kb"
    max_results: int  = Field(default=3, ge=1, le=20)


class QueryResponse(BaseModel):
    trace_id:      str
    response:      str
    citations:     list[Citation]
    quality_score: float
    pipeline_state: PipelineState
    latency_ms:    float
    tokens_used:   int
