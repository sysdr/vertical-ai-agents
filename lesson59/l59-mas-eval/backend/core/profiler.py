"""
Async-safe agent profiler.
Wraps coroutines with monotonic-clock measurement without blocking.
"""
from __future__ import annotations
import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Any

_records: dict[str, list[int]] = defaultdict(list)   # agent → [duration_ns, ...]


def profile_async(agent_name: str):
    """Decorator: measure wall time of an async function per agent."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic_ns()
            try:
                return await fn(*args, **kwargs)
            finally:
                _records[agent_name].append(time.monotonic_ns() - start)
        return wrapper
    return decorator


class AgentProfiler:
    """Collects and exposes per-agent timing statistics."""

    @staticmethod
    def record(agent_name: str, fn_name: str, duration_ns: int) -> None:
        _records[f"{agent_name}.{fn_name}"].append(duration_ns)

    @staticmethod
    def snapshot() -> dict[str, dict[str, float]]:
        result = {}
        for key, durations in _records.items():
            if not durations:
                continue
            result[key] = {
                "count": len(durations),
                "total_ns": sum(durations),
                "mean_ns": sum(durations) / len(durations),
                "max_ns": max(durations),
                "min_ns": min(durations),
            }
        return result

    @staticmethod
    def reset() -> None:
        _records.clear()

    @staticmethod
    def instrument_tool(agent_name: str):
        """Wrap a sync/async callable with profiling."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.monotonic_ns()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                finally:
                    AgentProfiler.record(agent_name, fn.__name__, time.monotonic_ns() - start)
            return async_wrapper
        return decorator
