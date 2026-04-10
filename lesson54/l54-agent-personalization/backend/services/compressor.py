"""
ContextCompressor — budget-aware personalization context builder.
Integrates with L53 BudgetEnforcer semantics.
"""
from __future__ import annotations
from ..models.profile import PreferenceVector, PersonalizationContext, PersonaType
from ..models.budget import BudgetState
from .router import AdaptiveAgentRouter

_router = AdaptiveAgentRouter()


def _rich_prefix(vector: PreferenceVector, persona: PersonaType) -> str:
    top = vector.top_dimensions(n=3)
    dims_desc = ", ".join([f"{k.replace('_',' ')}={v:.2f}" for k, v in top.items()])
    system = _router.get_system_prompt(persona)
    return (
        f"{system}\n\n"
        f"[User calibration — top preferences: {dims_desc}. "
        f"Adjust depth, formality, and format accordingly.]"
    )


def _standard_prefix(vector: PreferenceVector, persona: PersonaType) -> str:
    system = _router.get_system_prompt(persona)
    top = vector.top_dimensions(n=1)
    key, val = list(top.items())[0]
    hint = "high" if val > 0.7 else ("low" if val < 0.3 else "moderate")
    return f"{system}\n[User preference: {hint} {key.replace('_',' ')}.]"


def _minimal_prefix(persona: PersonaType) -> str:
    return f"[Persona: {persona.value}] {_router.get_system_prompt(persona)[:80]}"


class ContextCompressor:
    """
    Three-tier compression based on remaining budget fraction.
    Integrates L53 BudgetState for context-aware degradation.
    """

    def build_context(self, vector: PreferenceVector,
                      budget: BudgetState,
                      persona: PersonaType) -> PersonalizationContext:
        fraction = budget.remaining_fraction()

        if fraction > 0.6:
            prefix = _rich_prefix(vector, persona)
            return PersonalizationContext(
                tier="RICH", system_prefix=prefix, token_cost=85, persona=persona
            )
        elif fraction > 0.3:
            prefix = _standard_prefix(vector, persona)
            return PersonalizationContext(
                tier="STANDARD", system_prefix=prefix, token_cost=32, persona=persona
            )
        else:
            prefix = _minimal_prefix(persona)
            return PersonalizationContext(
                tier="MINIMAL", system_prefix=prefix, token_cost=8, persona=persona
            )

    def build_default(self, persona: PersonaType = PersonaType.DEFAULT) -> PersonalizationContext:
        budget = BudgetState()  # full budget
        vector = PreferenceVector.default()
        return self.build_context(vector, budget, persona)
