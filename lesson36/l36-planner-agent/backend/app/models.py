"""Pydantic models for query planning"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

class SubQuery(BaseModel):
    """Individual sub-query in decomposition plan"""
    text: str = Field(..., min_length=5, max_length=500, description="The sub-query text")
    rationale: str = Field(..., min_length=10, max_length=200, description="Why this sub-query helps")
    requires_previous: bool = Field(default=False, description="Whether this depends on previous results")

class QueryPlan(BaseModel):
    """Complete query decomposition plan"""
    original_query: str
    requires_decomposition: bool
    sub_queries: List[SubQuery] = Field(..., max_length=5)
    strategy: Literal["parallel", "sequential"] = "parallel"
    estimated_complexity: Literal["low", "medium", "high"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PlanRequest(BaseModel):
    """Request for query planning"""
    query: str = Field(..., min_length=3, max_length=1000)
    max_sub_queries: int = Field(default=5, ge=1, le=10)

class PlanResponse(BaseModel):
    """Response containing query plan"""
    success: bool
    plan: Optional[QueryPlan] = None
    error: Optional[str] = None
    processing_time_ms: float
