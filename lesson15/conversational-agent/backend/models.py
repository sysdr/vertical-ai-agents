from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class ConversationState(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    GOAL_SEEKING = "goal_seeking"
    COMPLETED = "completed"

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    
    def dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0)
        )

@dataclass
class Goal:
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completion_criteria: Optional[str] = None
    
    def dict(self) -> dict:
        return {
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "completed": self.completed,
            "completion_criteria": self.completion_criteria
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed=data.get("completed", False),
            completion_criteria=data.get("completion_criteria")
        )

@dataclass
class ConversationStateModel:
    user_id: str
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    active_goals: List[Goal] = field(default_factory=list)
    state: ConversationState = ConversationState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "messages": [m.dict() for m in self.messages],
            "active_goals": [g.dict() for g in self.active_goals],
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "total_tokens": self.total_tokens
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            user_id=data["user_id"],
            conversation_id=data["conversation_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            active_goals=[Goal.from_dict(g) for g in data.get("active_goals", [])],
            state=ConversationState(data.get("state", "initializing")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            total_tokens=data.get("total_tokens", 0)
        )
    
    def add_message(self, role: str, content: str, token_count: int = 0):
        msg = Message(role=role, content=content, token_count=token_count)
        self.messages.append(msg)
        self.total_tokens += token_count
        self.updated_at = datetime.now()
        if len(self.messages) == 1:
            self.state = ConversationState.ACTIVE
    
    def add_goal(self, description: str, criteria: Optional[str] = None):
        goal = Goal(description=description, completion_criteria=criteria)
        self.active_goals.append(goal)
        self.state = ConversationState.GOAL_SEEKING
        self.updated_at = datetime.now()
    
    def complete_goal(self, goal_description: str):
        for goal in self.active_goals:
            if goal.description == goal_description:
                goal.completed = True
                self.updated_at = datetime.now()
                break
        if all(g.completed for g in self.active_goals) and self.active_goals:
            self.state = ConversationState.COMPLETED
