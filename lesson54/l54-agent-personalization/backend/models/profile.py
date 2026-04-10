from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import numpy as np
import json
import base64


class PersonaType(str, Enum):
    EXECUTIVE   = "EXECUTIVE"
    PRACTITIONER = "PRACTITIONER"
    LEARNER     = "LEARNER"
    ANALYST     = "ANALYST"
    DEFAULT     = "DEFAULT"


@dataclass
class PreferenceVector:
    """
    8-dimensional preference encoding.
    Dimensions: [verbosity, formality, technical_depth, example_density,
                 hedge_tolerance, narrative_vs_list, domain_breadth, response_urgency]
    All values in [0.0, 1.0]
    """
    verbosity: float = 0.5
    formality: float = 0.5
    technical_depth: float = 0.5
    example_density: float = 0.5
    hedge_tolerance: float = 0.5
    narrative_vs_list: float = 0.5
    domain_breadth: float = 0.5
    response_urgency: float = 0.5

    @classmethod
    def default(cls) -> "PreferenceVector":
        return cls()

    def to_list(self) -> list[float]:
        return [
            self.verbosity, self.formality, self.technical_depth,
            self.example_density, self.hedge_tolerance, self.narrative_vs_list,
            self.domain_breadth, self.response_urgency
        ]

    def to_np(self) -> np.ndarray:
        return np.array(self.to_list(), dtype=np.float32)

    def dominant_persona(self) -> str:
        """Fast single-call persona label from dominant dimension."""
        dims = {
            "brief-formal": self.formality * (1 - self.verbosity),
            "deep-technical": self.technical_depth * self.example_density,
            "pedagogical": self.example_density * (1 - self.technical_depth),
            "analytical": self.hedge_tolerance * self.technical_depth,
        }
        label = max(dims, key=dims.get)
        return label

    def top_dimensions(self, n: int = 3) -> dict[str, float]:
        d = asdict(self)
        return dict(sorted(d.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:n])

    def add_laplace_noise(self, epsilon: float = 0.1) -> "PreferenceVector":
        """Differential privacy: Laplace noise for export safety."""
        rng = np.random.default_rng()
        noise = rng.laplace(0, 1/epsilon, 8) * 0.02  # small scale factor
        noisy = np.clip(self.to_np() + noise, 0.0, 1.0)
        fields = list(asdict(self).keys())
        return PreferenceVector(**dict(zip(fields, noisy.tolist())))

    def l2_distance(self, other: "PreferenceVector") -> float:
        return float(np.linalg.norm(self.to_np() - other.to_np()))

    def to_adk_metadata(self) -> dict[str, str]:
        """Google ADK UserContext metadata format (L55 prep)."""
        payload = json.dumps(asdict(self))
        return {
            "preference_vector_b64": base64.b64encode(payload.encode()).decode(),
            "dominant_persona": self.dominant_persona(),
            "schema_version": "1.0"
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreferenceVector":
        fields = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class UserProfile:
    user_id: str
    display_name: str = "Anonymous"
    email: str = ""
    consent_behavioral: bool = False
    consent_embedding: bool = False
    explicit_preferences: dict = field(default_factory=dict)
    preference_vector: Optional[PreferenceVector] = None
    persona: PersonaType = PersonaType.DEFAULT
    interaction_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_adk_user_context(self) -> dict:
        """L55 Google ADK UserContext proto-compatible dict."""
        meta = {}
        if self.preference_vector:
            meta.update(self.preference_vector.to_adk_metadata())
        meta["persona"] = self.persona.value
        meta["interaction_count"] = str(self.interaction_count)
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "metadata": meta
        }


@dataclass
class PersonalizationContext:
    tier: str  # RICH | STANDARD | MINIMAL
    system_prefix: str
    token_cost: int
    persona: PersonaType = PersonaType.DEFAULT
