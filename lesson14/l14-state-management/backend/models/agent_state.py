"""
Agent State Models - Pydantic schemas for type-safe state management
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class StateStatus(str, Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    AWAITING_INPUT = "AWAITING_INPUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Goal(BaseModel):
    """Individual goal within agent state"""
    id: str
    description: str
    status: str = "pending"
    priority: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    turn_id: int
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0
    context_summary: Optional[str] = None

class AgentState(BaseModel):
    """
    Core agent state model with versioning and persistence support.
    Used across all VAIA agent implementations.
    """
    session_id: str
    version: int = 1
    state_status: StateStatus = StateStatus.IDLE
    
    # Context and history
    user_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    context_summary: Optional[str] = None
    
    # Goals and planning
    current_goal: Optional[str] = None
    goals: List[Goal] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistics
    total_turns: int = 0
    total_tokens: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

class StateSnapshot(BaseModel):
    """Historical state snapshot for rollback"""
    snapshot_id: str
    session_id: str
    version: int
    state_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StateDiff(BaseModel):
    """Difference between two state versions"""
    from_version: int
    to_version: int
    changes: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
