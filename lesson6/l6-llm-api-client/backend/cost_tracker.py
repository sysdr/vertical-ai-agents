from typing import Dict
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class CostTracker:
    """Track API usage and costs in real-time."""
    
    def __init__(self, input_cost_per_1m: float, output_cost_per_1m: float):
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        
        # Usage counters
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Session tracking
        self.session_start = datetime.now()
        
        logger.info("cost_tracker_initialized",
                   input_cost=input_cost_per_1m,
                   output_cost=output_cost_per_1m)
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage and calculate cost."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        request_cost = input_cost + output_cost
        
        self.total_cost += request_cost
        
        logger.info("usage_recorded",
                   input_tokens=input_tokens,
                   output_tokens=output_tokens,
                   cost_usd=round(request_cost, 6))
        
        return request_cost
    
    def get_stats(self) -> Dict[str, any]:
        """Get usage statistics."""
        uptime = datetime.now() - self.session_start
        avg_cost_per_request = (self.total_cost / self.total_requests 
                                if self.total_requests > 0 else 0)
        
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_request": round(avg_cost_per_request, 6),
            "uptime_seconds": int(uptime.total_seconds()),
            "requests_per_hour": int(self.total_requests / (uptime.total_seconds() / 3600))
                if uptime.total_seconds() > 0 else 0
        }
    
    def reset_stats(self):
        """Reset all counters."""
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.session_start = datetime.now()
        
        logger.info("cost_tracker_reset")
