import tiktoken
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    """Production-grade token counting using tiktoken"""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize with model-specific encoding"""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self.model = model
            logger.info(f"TokenCounter initialized for model: {model}")
        except KeyError:
            # Fallback to cl100k_base (GPT-4, Gemini compatible)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = "cl100k_base"
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting error: {e}")
            # Rough fallback: 4 chars per token average
            return len(text) // 4
    
    def count_messages(self, messages: list) -> int:
        """Count tokens in message array (chat format)"""
        total = 0
        for message in messages:
            # Add message formatting overhead (role, delimiters)
            total += 4  # Overhead per message
            
            if isinstance(message, dict):
                for key, value in message.items():
                    total += self.count(str(value))
            else:
                total += self.count(str(message))
        
        total += 2  # Overall conversation overhead
        return total
    
    def analyze(self, text: str, max_tokens: int = 30000) -> Dict:
        """Analyze token usage with recommendations"""
        token_count = self.count(text)
        usage_percent = (token_count / max_tokens) * 100
        
        # Determine recommendation
        if usage_percent < 70:
            recommendation = "optimal"
            action = "none"
        elif usage_percent < 90:
            recommendation = "moderate"
            action = "light_compression"
        else:
            recommendation = "critical"
            action = "aggressive_compression"
        
        return {
            "token_count": token_count,
            "max_tokens": max_tokens,
            "usage_percent": round(usage_percent, 2),
            "remaining_tokens": max_tokens - token_count,
            "recommendation": recommendation,
            "action": action,
            "model": self.model
        }
    
    def estimate_cost(self, token_count: int, price_per_1k: float = 0.01) -> float:
        """Estimate API cost based on token count"""
        return (token_count / 1000) * price_per_1k

# Global instance
token_counter = TokenCounter()
