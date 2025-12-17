import time
import asyncio
from enum import Enum
import structlog

logger = structlog.get_logger()

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = asyncio.Lock()
        
        logger.info("circuit_breaker_initialized",
                   failure_threshold=failure_threshold,
                   timeout=timeout)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    logger.info("circuit_breaker_half_open",
                              failures=self.failure_count)
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open. Retry after {self.timeout}s"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call."""
        async with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:  # Require 2 successes to close
                    logger.info("circuit_breaker_closed",
                              successes=self.success_count)
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    async def _record_failure(self):
        """Record failed call."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error("circuit_breaker_opened",
                           failures=self.failure_count,
                           timeout=self.timeout)
                self.state = CircuitState.OPEN
                self.success_count = 0
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "threshold": self.failure_threshold
        }
