class AgentBudgetAllocator:
    """Distributes token budget across multiple agents"""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.allocations = {
            "planner": int(total_budget * 0.30),      # 30% for planning
            "retriever": int(total_budget * 0.20),    # 20% for retrieval
            "validator": int(total_budget * 0.15),    # 15% for validation
            "synthesizer": int(total_budget * 0.35)   # 35% for synthesis
        }
        self.used = {agent: 0 for agent in self.allocations}
    
    def reserve(self, agent: str, required: int) -> bool:
        """Check if budget available for agent"""
        if agent not in self.allocations:
            return False
        
        available = self.allocations[agent] - self.used[agent]
        if required <= available:
            self.used[agent] += required
            return True
        return False
    
    def get_remaining(self, agent: str) -> int:
        """Get remaining budget for agent"""
        if agent not in self.allocations:
            return 0
        return self.allocations[agent] - self.used[agent]
    
    def get_usage_summary(self) -> dict:
        """Get budget usage summary"""
        return {
            "total_budget": self.total_budget,
            "allocations": self.allocations,
            "used": self.used,
            "remaining": {
                agent: self.allocations[agent] - self.used[agent]
                for agent in self.allocations
            }
        }

class IterationController:
    """Controls max iterations per agent (from L34)"""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.current_iteration = 0
    
    def next_iteration(self) -> bool:
        """Increment and check if can continue"""
        self.current_iteration += 1
        return self.current_iteration <= self.max_iterations
    
    def is_exhausted(self) -> bool:
        """Check if iterations exhausted"""
        return self.current_iteration >= self.max_iterations
    
    def reset(self):
        """Reset counter"""
        self.current_iteration = 0
