import structlog
import logging
import sys
from datetime import datetime
from typing import Dict, Any

def setup_logging():
    """Configure structured JSON logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    return structlog.get_logger()

class TraceLogger:
    """Context-aware logger with trace ID propagation"""
    
    def __init__(self):
        self.logger = setup_logging()
    
    def log_stage(self, trace_id: str, stage: str, **kwargs):
        """Log pipeline stage with context"""
        self.logger.info(
            "pipeline_stage",
            trace_id=trace_id,
            stage=stage,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def log_error(self, trace_id: str, stage: str, error: Exception, **kwargs):
        """Log error with full context"""
        self.logger.error(
            "pipeline_error",
            trace_id=trace_id,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
    
    def log_metric(self, trace_id: str, metric_name: str, value: float, **kwargs):
        """Log individual metric"""
        self.logger.info(
            "metric",
            trace_id=trace_id,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )

logger = TraceLogger()
