import pytest
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure tests run in mock mode and never hit external APIs
os.environ.setdefault("USE_MOCK_GEMINI", "true")
os.environ.setdefault("GEMINI_API_KEY", "mock-api-key")

from backend.gemini_client import GeminiAPIClient
from backend.rate_limiter import RateLimiter
from backend.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from backend.cost_tracker import CostTracker

@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test rate limiter allows requests within limits."""
    limiter = RateLimiter(rpm=60, tpm=10000)
    
    # Should allow first request
    result = await limiter.acquire(100)
    assert result["allowed"] == True
    assert result["rpm_remaining"] == 59

@pytest.mark.asyncio
async def test_rate_limiter_throttling():
    """Test rate limiter throttles when limits exceeded."""
    limiter = RateLimiter(rpm=2, tpm=100)
    
    # Use up all tokens
    await limiter.acquire(50)
    await limiter.acquire(50)
    
    # Next request should be throttled
    result = await limiter.acquire(10)
    assert result["allowed"] == False
    assert "wait_seconds" in result

@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1)
    
    async def failing_func():
        raise Exception("Simulated failure")
    
    # Trigger failures
    for _ in range(3):
        try:
            await breaker.call(failing_func)
        except:
            pass
    
    # Circuit should be open now
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(failing_func)

@pytest.mark.asyncio
async def test_cost_tracker():
    """Test cost tracking accuracy."""
    tracker = CostTracker(input_cost_per_1m=0.075, output_cost_per_1m=0.300)
    
    # Record usage
    cost = tracker.record_usage(input_tokens=1000, output_tokens=500)
    
    # Verify calculation
    expected_cost = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.300
    assert abs(cost - expected_cost) < 0.000001
    
    stats = tracker.get_stats()
    assert stats["total_requests"] == 1
    assert stats["total_input_tokens"] == 1000

@pytest.mark.asyncio
async def test_gemini_client_generate():
    """Test Gemini client generates content successfully."""
    client = GeminiAPIClient()
    
    result = await client.generate("Say hello in one word")
    
    assert result["success"] == True
    assert "text" in result
    assert result["cost_usd"] > 0
    assert result["latency_ms"] > 0

@pytest.mark.asyncio
async def test_gemini_client_stats():
    """Test client statistics collection."""
    client = GeminiAPIClient()
    
    await client.generate("Test prompt")
    
    stats = client.get_stats()
    assert "rate_limiter" in stats
    assert "circuit_breaker" in stats
    assert "cost_tracker" in stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
