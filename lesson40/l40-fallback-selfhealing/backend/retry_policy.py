import random
import asyncio
from typing import Optional

class RetryPolicy:
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, jitter: float = 0.3):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter_amount = delay * self.jitter * random.uniform(-1, 1)
        return max(0, delay + jitter_amount)
    
    async def wait(self, attempt: int):
        """Wait for calculated delay"""
        delay = self.get_delay(attempt)
        await asyncio.sleep(delay)

class AdaptiveRetryPolicy(RetryPolicy):
    """Adaptive retry policy based on failure type"""
    
    def get_delay(self, attempt: int, failure_type: str = "unknown") -> float:
        if failure_type == "compliance_violation":
            # Immediate retry for compliance issues
            return 0.1
        elif failure_type == "factual_inconsistency":
            # Standard exponential backoff
            return super().get_delay(attempt)
        elif failure_type == "insufficient_context":
            # Longer delay for context gathering
            return super().get_delay(attempt) * 1.5
        else:
            return super().get_delay(attempt)
