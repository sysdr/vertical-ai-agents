"""
@trace_step: non-invasive instrumentation decorator.
Wraps any async agent method without modifying agent logic.
"""
import functools
from typing import Callable, Any
from backend.tracing.context import get_current_trace_or_none, Span


def trace_step(
    agent_name: str,
    capture_input_fields: list[str] | None = None,
    capture_output_fields: list[str] | None = None,
):
    """
    Decorator factory. Usage:

        @trace_step("planner", capture_input_fields=["query"])
        async def plan(self, query: str) -> PlanResult:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = get_current_trace_or_none()
            if ctx is None:
                # Tracing disabled — run normally
                return await fn(*args, **kwargs)

            span: Span = ctx.start_span(agent_name)

            # Capture input summary (first positional arg that isn't self)
            if capture_input_fields:
                for i, name in enumerate(capture_input_fields):
                    idx = i + 1  # skip self
                    if idx < len(args):
                        val = args[idx]
                        span.input_summary[name] = str(val)[:512]
                    elif name in kwargs:
                        span.input_summary[name] = str(kwargs[name])[:512]

            try:
                result = await fn(*args, **kwargs)

                # Agents return objects with .to_trace_dict() or we do best-effort
                if hasattr(result, "to_trace_dict"):
                    trace_meta = result.to_trace_dict()
                    span.finish(
                        output=trace_meta.get("output_summary", {}),
                        decision=trace_meta.get("decision", ""),
                        status="success",
                        risk_score=trace_meta.get("risk_score", 0.0),
                        confidence=trace_meta.get("confidence", 0.0),
                        input_tokens=trace_meta.get("input_tokens", 0),
                        output_tokens=trace_meta.get("output_tokens", 0),
                        model_id=trace_meta.get("model_id", ""),
                        compliance_flags=trace_meta.get("compliance_flags", []),
                    )
                else:
                    span.finish(
                        output={"result": str(result)[:256]},
                        status="success",
                    )
                return result

            except Exception as exc:
                span.finish(status="error", error=str(exc)[:512])
                raise

        return wrapper
    return decorator
