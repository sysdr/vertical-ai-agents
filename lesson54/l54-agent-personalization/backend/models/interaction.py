from dataclasses import dataclass
from typing import Optional


@dataclass
class Interaction:
    interaction_id: str
    user_id: str
    user_msg: str
    agent_msg: str
    persona_used: str
    context_tier: str
    token_cost: int
    created_at: str
    session_id: str = ""
    feedback_score: Optional[float] = None


@dataclass
class InteractionFeedback:
    interaction_id: str
    score: float   # -1.0 to 1.0 implicit/explicit
    signal_type: str  # "explicit" | "implicit_positive" | "implicit_negative"
