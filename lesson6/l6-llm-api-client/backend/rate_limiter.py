import time
import asyncio
from typing import Dict
import structlog

logger = structlog.get_logger()

class RateLimiter:
    """Token bucket algorithm for RPM and TPM rate limiting."""
    
    def __init__(self, rpm: int, tpm: int):
        self.rpm_capacity = rpm
        self.tpm_capacity = tpm
        self.rpm_tokens = float(rpm)
        self.tpm_tokens = float(tpm)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
        logger.info("rate_limiter_initialized", 
                   rpm_capacity=rpm, tpm_capacity=tpm)
    
    async def acquire(self, token_count: int = 1) -> Dict[str, any]:
        """
        Attempt to acquire tokens for a request.
        Returns dict with success status and wait time if throttled.
        """
        async with self.lock:
            self._refill()
            
            if self.rpm_tokens >= 1 and self.tpm_tokens >= token_count:
                self.rpm_tokens -= 1
                self.tpm_tokens -= token_count
                
                logger.debug("rate_limit_acquired",
                           rpm_remaining=int(self.rpm_tokens),
                           tpm_remaining=int(self.tpm_tokens))
                
                return {
                    "allowed": True,
                    "rpm_remaining": int(self.rpm_tokens),
                    "tpm_remaining": int(self.tpm_tokens)
                }
            else:
                # Calculate wait time for token replenishment
                wait_time = self._calculate_wait_time(token_count)
                
                logger.warning("rate_limit_exceeded",
                             rpm_remaining=int(self.rpm_tokens),
                             tpm_remaining=int(self.tpm_tokens),
                             wait_seconds=wait_time)
                
                return {
                    "allowed": False,
                    "wait_seconds": wait_time,
                    "rpm_remaining": int(self.rpm_tokens),
                    "tpm_remaining": int(self.tpm_tokens)
                }
    
    def _refill(self):
        """Continuously refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Refill RPM tokens
        rpm_refill = (self.rpm_capacity / 60.0) * elapsed
        self.rpm_tokens = min(self.rpm_capacity, 
                             self.rpm_tokens + rpm_refill)
        
        # Refill TPM tokens
        tpm_refill = (self.tpm_capacity / 60.0) * elapsed
        self.tpm_tokens = min(self.tpm_capacity,
                             self.tpm_tokens + tpm_refill)
        
        self.last_refill = now
    
    def _calculate_wait_time(self, token_count: int) -> float:
        """Calculate seconds to wait for token replenishment."""
        if self.rpm_tokens < 1:
            rpm_wait = (1 - self.rpm_tokens) / (self.rpm_capacity / 60.0)
        else:
            rpm_wait = 0
        
        if self.tpm_tokens < token_count:
            tpm_wait = (token_count - self.tpm_tokens) / (self.tpm_capacity / 60.0)
        else:
            tpm_wait = 0
        
        return max(rpm_wait, tpm_wait)
    
    def get_stats(self) -> Dict[str, any]:
        """Get current rate limiter statistics."""
        return {
            "rpm_available": int(self.rpm_tokens),
            "rpm_capacity": self.rpm_capacity,
            "tpm_available": int(self.tpm_tokens),
            "tpm_capacity": self.tpm_capacity,
            "utilization_rpm": 1 - (self.rpm_tokens / self.rpm_capacity),
            "utilization_tpm": 1 - (self.tpm_tokens / self.tpm_capacity)
        }
