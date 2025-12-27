from typing import Dict, List, Optional
import logging
from utils.token_counter import token_counter
from services.summarizer import summarizer
import os

logger = logging.getLogger(__name__)

class ContextOptimizer:
    """Orchestrates full context optimization pipeline"""
    
    def __init__(self):
        self.max_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))
        self.compression_timeout = int(os.getenv("COMPRESSION_TIMEOUT_MS", "500")) / 1000
        logger.info(f"ContextOptimizer initialized (max_tokens={self.max_tokens})")
    
    async def optimize(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_quality: bool = True
    ) -> Dict:
        """
        Optimize context with intelligent strategy selection
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Step 1: Analyze current state
        analysis = token_counter.analyze(text, max_tokens)
        
        # Step 2: Decide on strategy
        if analysis["recommendation"] == "optimal":
            # No compression needed
            return {
                "optimized_text": text,
                "original_tokens": analysis["token_count"],
                "optimized_tokens": analysis["token_count"],
                "compression_ratio": 1.0,
                "strategy": "none",
                "action_taken": "pass_through",
                "quality_preserved": True,
                "cost_savings": 0
            }
        
        # Step 3: Select compression strategy
        if analysis["recommendation"] == "moderate":
            strategy = "extractive"
            target_ratio = 0.7  # Light compression
        else:  # critical
            strategy = "hybrid"
            target_ratio = 0.5  # Aggressive compression
        
        # Step 4: Compress
        try:
            result = await summarizer.summarize(
                text,
                strategy=strategy,
                target_ratio=target_ratio
            )
            
            # Step 5: Validate compression quality
            compressed_tokens = token_counter.count(result["summary"])
            
            # Check if compression helped
            if compressed_tokens >= analysis["token_count"] * 0.9:
                # Compression didn't help enough, truncate instead
                logger.warning("Compression ineffective, truncating")
                truncated = self._truncate_text(text, max_tokens)
                truncated_tokens = token_counter.count(truncated)
                
                return {
                    "optimized_text": truncated,
                    "original_tokens": analysis["token_count"],
                    "optimized_tokens": truncated_tokens,
                    "compression_ratio": analysis["token_count"] / truncated_tokens,
                    "strategy": "truncation",
                    "action_taken": "compression_failed_truncated",
                    "quality_preserved": False,
                    "cost_savings": self._calculate_savings(
                        analysis["token_count"], 
                        truncated_tokens
                    )
                }
            
            # Compression succeeded
            return {
                "optimized_text": result["summary"],
                "original_tokens": analysis["token_count"],
                "optimized_tokens": compressed_tokens,
                "compression_ratio": result["compression_ratio"],
                "strategy": strategy,
                "action_taken": "compressed",
                "quality_preserved": preserve_quality,
                "compression_time_ms": result["compression_time_ms"],
                "cost_savings": self._calculate_savings(
                    analysis["token_count"],
                    compressed_tokens
                )
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            # Fallback to truncation
            truncated = self._truncate_text(text, max_tokens)
            truncated_tokens = token_counter.count(truncated)
            
            return {
                "optimized_text": truncated,
                "original_tokens": analysis["token_count"],
                "optimized_tokens": truncated_tokens,
                "compression_ratio": analysis["token_count"] / truncated_tokens,
                "strategy": "truncation",
                "action_taken": "error_truncated",
                "quality_preserved": False,
                "error": str(e),
                "cost_savings": self._calculate_savings(
                    analysis["token_count"],
                    truncated_tokens
                )
            }
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Intelligently truncate text to fit token limit"""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        
        if last_period > max_chars * 0.8:  # If period is in last 20%
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _calculate_savings(self, original_tokens: int, optimized_tokens: int) -> Dict:
        """Calculate cost savings from optimization"""
        tokens_saved = original_tokens - optimized_tokens
        cost_per_1k = 0.01  # $0.01 per 1K tokens (example pricing)
        
        money_saved = (tokens_saved / 1000) * cost_per_1k
        
        return {
            "tokens_saved": tokens_saved,
            "percent_reduction": round((tokens_saved / original_tokens) * 100, 2),
            "money_saved_per_request": round(money_saved, 6),
            "projected_monthly_savings": round(money_saved * 1000000, 2)  # At 1M requests/month
        }

# Global instance
context_optimizer = ContextOptimizer()
