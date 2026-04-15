"""
DebugSession: state machine managing trace recording and analysis.
States: IDLE → ATTACHED → RECORDING → ANALYZING → DONE
"""
from __future__ import annotations
import asyncio
import time
import uuid
from enum import Enum
from typing import Optional

from models.trace import TraceRecord, EventType
from core.logger import MASTraceLogger
from core.profiler import AgentProfiler


class SessionState(str, Enum):
    IDLE       = "IDLE"
    ATTACHED   = "ATTACHED"
    RECORDING  = "RECORDING"
    ANALYZING  = "ANALYZING"
    DONE       = "DONE"
    ERROR      = "ERROR"


class DebugSession:
    IDLE_TIMEOUT_S = 30.0

    def __init__(self, tracer: MASTraceLogger):
        self._tracer = tracer
        self.session_id: str = str(uuid.uuid4())
        self.trace_id: str = str(uuid.uuid4())
        self.state: SessionState = SessionState.IDLE
        self.records: list[TraceRecord] = []
        self._last_event_ts: float = 0.0
        self._timeout_task: Optional[asyncio.Task] = None
        self.final_output: str = ""
        self.error: Optional[str] = None
        self._attached_agents: list[str] = []

    async def attach(self, agent_names: list[str]) -> None:
        assert self.state == SessionState.IDLE, "Already attached"
        self.state = SessionState.ATTACHED
        self._attached_agents = agent_names
        AgentProfiler.reset()
        rec = TraceRecord(
            trace_id=self.trace_id,
            agent_name="session",
            event_type="SESSION_START",
            payload={"agents": agent_names},
        )
        await self._tracer.emit(rec)

    async def emit(self, record: TraceRecord) -> None:
        record.trace_id = self.trace_id
        self.records.append(record)
        self._last_event_ts = time.monotonic()
        if self.state == SessionState.ATTACHED:
            self.state = SessionState.RECORDING
            self._schedule_timeout()
        await self._tracer.emit(record)

    async def close(self) -> None:
        if self._timeout_task:
            self._timeout_task.cancel()
        if self.state in (SessionState.RECORDING, SessionState.ATTACHED):
            self.state = SessionState.ANALYZING

    def _schedule_timeout(self) -> None:
        if self._timeout_task:
            self._timeout_task.cancel()
        self._timeout_task = asyncio.create_task(self._idle_watchdog())

    async def _idle_watchdog(self) -> None:
        while True:
            await asyncio.sleep(5)
            idle = time.monotonic() - self._last_event_ts
            if idle >= self.IDLE_TIMEOUT_S and self.state == SessionState.RECORDING:
                self.state = SessionState.ANALYZING
                break

    def reset(self) -> None:
        self.state = SessionState.IDLE
        self.records = []
        self.trace_id = str(uuid.uuid4())
        self.final_output = ""
        self.error = None
        if self._timeout_task:
            self._timeout_task.cancel()
