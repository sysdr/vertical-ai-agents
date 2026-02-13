import time
import asyncio
from typing import Optional
from models import AgentResponse, RetryMetrics
from circuit_breaker import CircuitBreaker
from retry_policy import AdaptiveRetryPolicy
from planner_agent import EnhancedPlannerAgent
from retriever_agent import RetrieverAgent
from validator_agent import EnhancedValidatorAgent

class FallbackOrchestrator:
    def __init__(self, api_key: str):
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=30)
        self.retry_policy = AdaptiveRetryPolicy(max_attempts=3, base_delay=1.0)
        
        self.planner = EnhancedPlannerAgent(api_key)
        self.retriever = RetrieverAgent(api_key)
        self.validator = EnhancedValidatorAgent(api_key)
        
        self.retry_history = []
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
    
    async def execute_with_fallback(self, query: str, simulate_failure: bool = False) -> AgentResponse:
        """Execute agent pipeline with fallback logic"""
        start_time = time.time()
        self.total_queries += 1
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.failed_queries += 1
            return AgentResponse(
                success=False,
                error="Circuit breaker open - system in recovery mode",
                metadata={"circuit_state": self.circuit_breaker.get_state()}
            )
        
        retry_count = 0
        current_query = query
        
        for attempt in range(self.retry_policy.max_attempts):
            try:
                # Execute agent pipeline
                plan = await self.planner.plan(current_query)
                context = await self.retriever.retrieve(plan)
                validation = await self.validator.validate(context, simulate_failure)
                
                # Track retry metrics
                retry_metric = RetryMetrics(
                    attempt_number=attempt,
                    delay_seconds=0 if attempt == 0 else self.retry_policy.get_delay(attempt - 1),
                    reformulation_applied=current_query != query,
                    validation_result=validation,
                    timestamp=time.time()
                )
                self.retry_history.append(retry_metric)
                
                if validation.success:
                    self.circuit_breaker.record_success()
                    self.successful_queries += 1
                    total_latency = (time.time() - start_time) * 1000
                    
                    return AgentResponse(
                        success=True,
                        data={
                            "query": current_query,
                            "context": context.model_dump(),
                            "validation": validation.model_dump()
                        },
                        retry_count=retry_count,
                        total_latency_ms=total_latency,
                        metadata={
                            "attempts": attempt + 1,
                            "reformulations": retry_count
                        }
                    )
                
                # Validation failed - reformulate and retry
                retry_count += 1
                if attempt < self.retry_policy.max_attempts - 1:
                    reformulated_plan = await self.planner.reformulate(query, validation.failure_signal)
                    current_query = reformulated_plan.reformulated_query or query
                    
                    # Wait with backoff
                    delay = self.retry_policy.get_delay(attempt, validation.failure_signal.type.value)
                    await asyncio.sleep(delay)
                
            except Exception as e:
                self.circuit_breaker.record_failure()
                if attempt == self.retry_policy.max_attempts - 1:
                    self.failed_queries += 1
                    return AgentResponse(
                        success=False,
                        error=f"Pipeline failed: {str(e)}",
                        retry_count=retry_count,
                        total_latency_ms=(time.time() - start_time) * 1000
                    )
        
        # Max retries exhausted - graceful degradation
        self.circuit_breaker.record_failure()
        self.failed_queries += 1
        return self.graceful_degradation(query, retry_count)
    
    def graceful_degradation(self, query: str, retry_count: int) -> AgentResponse:
        """Provide graceful degradation response"""
        return AgentResponse(
            success=False,
            error="Max retries exhausted",
            retry_count=retry_count,
            metadata={
                "message": "Unable to validate response after multiple attempts",
                "suggestion": "Try rephrasing your query or contact support"
            }
        )
    
    def get_metrics(self) -> dict:
        """Get orchestrator metrics"""
        return {
            "circuit_breaker": self.circuit_breaker.get_state(),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "retry_history_count": len(self.retry_history),
            "recent_retries": [m.model_dump() for m in self.retry_history[-10:]]
        }
