"""
L43: Traceability Layer — re-uses L42 AuditLogger patterns.
Writes structured JSON audit records keyed by trace_id.
"""
from __future__ import annotations
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from .models import TraceEvent, PipelineState
import time


AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit_logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class PipelineTracer:
    """
    Accumulates TraceEvents for a single pipeline execution.
    Writes a consolidated audit record on finalize().
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self._events:  list[TraceEvent] = []
        self._started: float = time.time()

    async def log(self, event: TraceEvent) -> None:
        self._events.append(event)

    async def finalize(
        self,
        final_state: PipelineState,
        result: Optional[dict] = None
    ) -> str:
        audit = {
            "trace_id":    self.trace_id,
            "start_time":  self._started,
            "end_time":    time.time(),
            "duration_ms": round((time.time() - self._started) * 1000, 2),
            "final_state": final_state.value,
            "events":      [e.model_dump(mode="json") for e in self._events],
            "result":      result or {},
        }
        audit_file = AUDIT_DIR / f"{self.trace_id}.json"
        try:
            audit_file.write_text(json.dumps(audit, indent=2))
        except Exception:
            pass  # do not block response if audit write fails
        return self.trace_id

    def get_events(self) -> list[TraceEvent]:
        return list(self._events)


class AuditStore:
    """Query completed audit records."""

    @staticmethod
    def get(trace_id: str) -> Optional[dict]:
        f = AUDIT_DIR / f"{trace_id}.json"
        return json.loads(f.read_text()) if f.exists() else None

    @staticmethod
    def list_recent(n: int = 20) -> list[dict]:
        files = sorted(AUDIT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        result = []
        for f in files[:n]:
            try:
                result.append(json.loads(f.read_text()))
            except Exception:
                pass
        return result
