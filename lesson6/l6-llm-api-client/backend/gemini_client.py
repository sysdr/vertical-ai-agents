import google.generativeai as genai
import asyncio
import time
import random
from typing import Dict
import structlog
from config.settings import settings
from backend.rate_limiter import RateLimiter
from backend.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from backend.cost_tracker import CostTracker

logger = structlog.get_logger()

class GeminiAPIClient:
    """Production-grade Gemini API client with rate limiting, circuit breaker, cost tracking, and mock mode."""
    
    def __init__(self):
        self.use_mock = settings.use_mock_gemini

        # Configure Gemini when not in mock mode
        if not self.use_mock:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        else:
            self.model = None
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            rpm=settings.rate_limit_rpm,
            tpm=settings.rate_limit_tpm
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout
        )
        self.cost_tracker = CostTracker(
            input_cost_per_1m=settings.cost_input_token,
            output_cost_per_1m=settings.cost_output_token
        )
        
        logger.info("gemini_client_initialized", model=settings.gemini_model, use_mock=self.use_mock)
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, any]:
        """
        Generate content with full production patterns:
        - Rate limiting
        - Circuit breaker
        - Retry with exponential backoff
        - Cost tracking
        - Structured logging
        - Optional mock mode for offline/demo use
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        logger.info("request_started", request_id=request_id, prompt_length=len(prompt), use_mock=self.use_mock)
        
        try:
            # Estimate tokens (rough approximation: 4 chars per token)
            estimated_tokens = max(1, len(prompt) // 4)
            
            # Rate limiting
            rate_limit_result = await self.rate_limiter.acquire(estimated_tokens)
            if not rate_limit_result["allowed"]:
                logger.warning("request_throttled", 
                             request_id=request_id,
                             wait_seconds=rate_limit_result["wait_seconds"])
                return {
                    "success": False,
                    "error": "rate_limited",
                    "wait_seconds": rate_limit_result["wait_seconds"],
                    "request_id": request_id
                }
            
            # Execute generation (mock or real)
            if self.use_mock:
                response = await self._mock_generate_response(prompt, **kwargs)
            else:
                response = await self.circuit_breaker.call(
                    self._generate_with_retry,
                    prompt,
                    **kwargs
                )
            
            # Track cost
            input_tokens = response.get("input_tokens", estimated_tokens)
            output_tokens = response.get("output_tokens", max(1, len(response.get("text", "")) // 4))
            cost = self.cost_tracker.record_usage(input_tokens, output_tokens)
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("request_completed",
                       request_id=request_id,
                       latency_ms=latency_ms,
                       cost_usd=round(cost, 6),
                       use_mock=self.use_mock)
            
            return {
                "success": True,
                "text": response["text"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
                "latency_ms": latency_ms,
                "request_id": request_id
            }
            
        except CircuitBreakerOpenError as e:
            logger.error("circuit_breaker_blocked", request_id=request_id)
            return {
                "success": False,
                "error": "circuit_breaker_open",
                "message": str(e),
                "request_id": request_id
            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("request_failed",
                        request_id=request_id,
                        error=str(e),
                        latency_ms=latency_ms)
            return {
                "success": False,
                "error": "api_error",
                "message": str(e),
                "request_id": request_id
            }
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> Dict:
        """Generate with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                # Run synchronous API call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(prompt, **kwargs)
                )
                
                return {
                    "text": response.text,
                    "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0)
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                base_delay = 0.1 * (2 ** attempt)
                jitter = base_delay * random.random()
                wait_time = base_delay + jitter
                
                logger.warning("retry_attempt",
                             attempt=attempt + 1,
                             max_retries=max_retries,
                             wait_seconds=round(wait_time, 2),
                             error=str(e))
                
                await asyncio.sleep(wait_time)
    
    async def _mock_generate_response(self, prompt: str, **kwargs) -> Dict[str, any]:
        """Mock response for offline/testing/demo runs."""
        await asyncio.sleep(0.05 + random.random() * 0.05)
        output_tokens = max(24, min(256, len(prompt) // 3 + 40))
        input_tokens = max(16, len(prompt) // 4 or 16)
        mock_text = f"[MOCK] Generated response for: {prompt[:80] or 'empty prompt'}"
        
        logger.debug("mock_response_generated",
                     output_tokens=output_tokens,
                     input_tokens=input_tokens)
        
        return {
            "text": mock_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive client statistics."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "cost_tracker": self.cost_tracker.get_stats()
        }
