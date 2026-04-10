"""
AdaptiveAgentRouter — deterministic persona selection via cosine similarity.
No LLM calls in the routing path.
"""
from __future__ import annotations
import numpy as np
from ..models.profile import PersonaType, PreferenceVector

# 8-dim persona archetype vectors
# [verbosity, formality, technical_depth, example_density,
#  hedge_tolerance, narrative_vs_list, domain_breadth, response_urgency]
PERSONA_ARCHETYPES = {
    PersonaType.EXECUTIVE:    np.array([0.2, 0.9, 0.4, 0.2, 0.2, 0.9, 0.5, 0.9], dtype=np.float32),
    PersonaType.PRACTITIONER: np.array([0.6, 0.4, 0.9, 0.9, 0.2, 0.3, 0.7, 0.6], dtype=np.float32),
    PersonaType.LEARNER:      np.array([0.7, 0.5, 0.4, 0.9, 0.7, 0.2, 0.4, 0.3], dtype=np.float32),
    PersonaType.ANALYST:      np.array([0.6, 0.7, 0.8, 0.5, 0.9, 0.5, 0.8, 0.5], dtype=np.float32),
}

PERSONA_SYSTEM_PROMPTS = {
    PersonaType.EXECUTIVE: (
        "You are a concise, high-impact advisor. "
        "Lead with the bottom line. Use bullets. No hedging. Max 3 sentences per point. "
        "Assume C-suite context."
    ),
    PersonaType.PRACTITIONER: (
        "You are a hands-on technical expert. "
        "Be precise, include concrete code examples, skip basic explanations. "
        "Use production-grade terminology."
    ),
    PersonaType.LEARNER: (
        "You are a patient, encouraging teacher. "
        "Build up from first principles. Use analogies and worked examples. "
        "Check understanding at the end."
    ),
    PersonaType.ANALYST: (
        "You are a rigorous analyst. "
        "Quantify claims where possible. Flag uncertainty explicitly. "
        "Structure responses with clear reasoning chains."
    ),
    PersonaType.DEFAULT: (
        "You are a helpful, balanced AI assistant."
    ),
}


class AdaptiveAgentRouter:
    def select_persona(self, vector: PreferenceVector) -> PersonaType:
        """Cosine similarity against archetype vectors. O(1), no LLM."""
        v = vector.to_np()
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return PersonaType.DEFAULT

        scores = {}
        for ptype, archetype in PERSONA_ARCHETYPES.items():
            similarity = float(np.dot(v, archetype) / (norm_v * np.linalg.norm(archetype)))
            scores[ptype] = similarity

        return max(scores, key=scores.get)

    def get_system_prompt(self, persona: PersonaType) -> str:
        return PERSONA_SYSTEM_PROMPTS.get(persona, PERSONA_SYSTEM_PROMPTS[PersonaType.DEFAULT])

    def score_all(self, vector: PreferenceVector) -> dict[str, float]:
        v = vector.to_np()
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return {p.value: 0.25 for p in PersonaType if p != PersonaType.DEFAULT}
        return {
            ptype.value: round(float(
                np.dot(v, archetype) / (norm_v * np.linalg.norm(archetype))
            ), 4)
            for ptype, archetype in PERSONA_ARCHETYPES.items()
        }
