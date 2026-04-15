"""
Pydantic models for MAS trace records.
Bridges L58 MultiModalMessage schema.
"""
from __future__ import annotations
import uuid
import time
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class TokenCost(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


EventType = Literal[
    "SEND", "RECEIVE", "LLM_CALL", "TOOL_CALL",
    "ROUTING_DECISION", "ERROR", "SESSION_START", "SESSION_END"
]

Modality = Literal["text", "image", "audio", "multimodal"]


class TraceRecord(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    agent_name: str
    event_type: EventType
    timestamp: float = Field(default_factory=time.monotonic)
    duration_ns: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    token_cost: TokenCost = Field(default_factory=TokenCost)
    modality: Modality = "text"
    error_msg: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "orchestrator",
                "event_type": "LLM_CALL",
                "modality": "text",
                "duration_ns": 1_200_000,
            }
        }


class TestCase(BaseModel):
    name: str
    expected_output_pattern: str = ""
    max_tokens: int = 50_000
    max_latency_ms: int = 30_000
    description: str = ""


class TestResult(BaseModel):
    test_name: str
    correctness: bool
    efficiency: bool
    topology_ok: Optional[bool] = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    test_results: list[TestResult]
    agent_hotspots: dict[str, int]   # agent_name → total_duration_ns
    critical_path: list[str]          # span_ids
    total_tokens: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total else 0.0
