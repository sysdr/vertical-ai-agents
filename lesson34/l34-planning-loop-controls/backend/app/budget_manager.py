from typing import Dict
from datetime import datetime

class BudgetException(Exception):
    """Base exception for budget violations."""
    pass

class IterationLimitException(BudgetException):
    """Raised when iteration limit is exceeded."""
    pass

class TokenBudgetException(BudgetException):
    """Raised when token budget is exceeded."""
    pass

class BudgetManager:
    """Manages iteration and token budgets for agent execution."""
    
    def __init__(self, policy: Dict[str, int]):
        self.max_iterations = policy['max_iterations']
        self.max_tokens = policy['max_tokens_per_turn']
        self.current_iteration = 0
        self.tokens_used = 0
        self.token_history = []
        self.start_time = datetime.now()
    
    def check_limits(self):
        """Check if current usage exceeds limits."""
        if self.current_iteration >= self.max_iterations:
            raise IterationLimitException(
                f"Exceeded max iterations: {self.current_iteration}/{self.max_iterations}"
            )
        if self.tokens_used >= self.max_tokens:
            raise TokenBudgetException(
                f"Exceeded token budget: {self.tokens_used}/{self.max_tokens}"
            )
    
    def record_llm_call(self, input_tokens: int, output_tokens: int, step_type: str):
        """Record tokens used in an LLM call."""
        total_tokens = input_tokens + output_tokens
        self.tokens_used += total_tokens
        self.current_iteration += 1
        
        self.token_history.append({
            'iteration': self.current_iteration,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'step_type': step_type,
            'cumulative_tokens': self.tokens_used
        })
        
        # Check limits after recording
        self.check_limits()
    
    def get_usage_report(self) -> Dict:
        """Generate detailed usage report."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'total_iterations': self.current_iteration,
            'total_tokens': self.tokens_used,
            'max_iterations': self.max_iterations,
            'max_tokens': self.max_tokens,
            'iteration_utilization': self.current_iteration / self.max_iterations if self.max_iterations > 0 else 0,
            'token_utilization': self.tokens_used / self.max_tokens if self.max_tokens > 0 else 0,
            'elapsed_seconds': elapsed_time,
            'tokens_per_second': self.tokens_used / elapsed_time if elapsed_time > 0 else 0,
            'token_history': self.token_history
        }
    
    def is_approaching_limit(self, threshold: float = 0.8) -> Dict[str, bool]:
        """Check if approaching budget limits."""
        return {
            'iteration_warning': self.current_iteration / self.max_iterations >= threshold,
            'token_warning': self.tokens_used / self.max_tokens >= threshold
        }
