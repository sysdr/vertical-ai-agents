from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    session_id: str
    query: str
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    response: str
    is_ambiguous: bool
    ambiguity_score: float
    clarification_questions: List[str] = []
    retrieved_documents: List[Dict[str, Any]] = []
    context_used: List[Message] = []
    conversation_state: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    total_tokens: int
    created_at: datetime
    last_activity: datetime
    state: str

class HealthResponse(BaseModel):
    redis: bool
    rag: bool
    chroma: bool
