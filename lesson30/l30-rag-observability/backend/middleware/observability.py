import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from services.metrics_collector import metrics
from services.structured_logger import logger

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Automatic request instrumentation and tracing"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate trace ID
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        
        # Start timing
        start_time = time.perf_counter()
        await metrics.start_trace(trace_id)
        
        logger.log_stage(
            trace_id=trace_id,
            stage="received",
            method=request.method,
            path=request.url.path
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            await metrics.end_trace(trace_id)
            
            logger.log_stage(
                trace_id=trace_id,
                stage="completed",
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            logger.log_error(
                trace_id=trace_id,
                stage="failed",
                error=e,
                duration_ms=duration_ms
            )
            
            metrics.increment("errors", {"error_type": type(e).__name__})
            raise
