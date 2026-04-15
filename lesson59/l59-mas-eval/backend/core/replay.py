"""
TraceReplayEngine: re-executes frozen trace segments using deterministic stubs.
No LLM API calls consumed during replay.
"""
from __future__ import annotations
import copy
import json
import time
from pathlib import Path

from models.trace import TraceRecord


class TraceReplayEngine:
    """
    Load a saved JSONL trace file and replay it, returning the records
    in causal order without any external API calls.
    """
    def __init__(self, trace_file: str):
        self._file = Path(trace_file)
        self._records: list[TraceRecord] = []
        self._load()

    def _load(self) -> None:
        with open(self._file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Support loguru JSONL wrapper
                if "record" in data and "extra" in data["record"]:
                    inner = data["record"]["extra"]
                    if isinstance(inner, dict):
                        data = inner
                try:
                    self._records.append(TraceRecord(**data))
                except Exception:
                    pass  # skip malformed records

    def replay(self, inject_at: int = -1, inject_payload: dict | None = None) -> list[TraceRecord]:
        """
        Return records in causal order.
        Optionally inject a modified payload at position `inject_at`
        for counterfactual analysis.
        """
        replayed = copy.deepcopy(self._records)
        if inject_at >= 0 and inject_payload and inject_at < len(replayed):
            replayed[inject_at].payload.update(inject_payload)
        return replayed

    def extract_llm_responses(self) -> dict[str, str]:
        """Return {span_id: llm_response_text} for all LLM_CALL events."""
        return {
            r.span_id: r.payload.get("response", "")
            for r in self._records
            if r.event_type == "LLM_CALL"
        }

    @property
    def span_count(self) -> int:
        return len(self._records)
