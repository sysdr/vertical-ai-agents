from dataclasses import dataclass


@dataclass
class BudgetState:
    """Imported from L53 BudgetEnforcer — used for context compression decisions."""
    max_tokens: int = 8000
    used_tokens: int = 0
    max_steps: int = 20
    used_steps: int = 0
    max_time_seconds: float = 60.0
    elapsed_seconds: float = 0.0

    def remaining_fraction(self) -> float:
        token_fraction = 1.0 - (self.used_tokens / self.max_tokens)
        step_fraction  = 1.0 - (self.used_steps  / self.max_steps)
        time_fraction  = 1.0 - (self.elapsed_seconds / self.max_time_seconds)
        return min(token_fraction, step_fraction, time_fraction)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)
