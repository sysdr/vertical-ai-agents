"""
InteractionGraph: reconstructs the causal DAG from trace records.
Uses networkx for O(V+E) critical path analysis.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any

import networkx as nx

from models.trace import TraceRecord


class InteractionGraph:
    def __init__(self, records: list[TraceRecord | dict]):
        self.G: nx.DiGraph = nx.DiGraph()
        self._build(records)

    def _build(self, records: list[Any]) -> None:
        span_set: set[str] = set()
        for r in records:
            if isinstance(r, dict):
                sid = r.get("span_id", "")
                pid = r.get("parent_id")
                data = {k: v for k, v in r.items() if k not in ("payload",)}
            else:
                sid = r.span_id
                pid = r.parent_id
                data = r.model_dump(exclude={"payload"})
                # Normalize nested token_cost for token accounting helpers.
                tc = data.get("token_cost") or {}
                if isinstance(tc, dict):
                    data["prompt_tok"] = tc.get("prompt_tokens", 0) or 0
                    data["compl_tok"] = tc.get("completion_tokens", 0) or 0

            self.G.add_node(sid, **data)
            span_set.add(sid)

            if pid and pid in span_set:
                dur = data.get("duration_ns", 0) or 0
                self.G.add_edge(pid, sid, weight=dur)

    def critical_path(self) -> list[str]:
        """Longest path by duration_ns — the real latency bottleneck."""
        if not self.G.nodes:
            return []
        try:
            return nx.dag_longest_path(self.G, weight="weight")
        except nx.NetworkXUnfeasible:
            return []

    def agent_hotspots(self) -> dict[str, int]:
        """Total duration per agent, descending."""
        totals: dict[str, int] = defaultdict(int)
        for _, data in self.G.nodes(data=True):
            name = data.get("agent_name", "unknown")
            dur = data.get("duration_ns", 0) or 0
            if dur > 0:
                totals[name] += dur
        return dict(sorted(totals.items(), key=lambda x: -x[1]))

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n, **{k: v for k, v in d.items() if k != "payload"}}
                for n, d in self.G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "weight": d.get("weight", 0)}
                for u, v, d in self.G.edges(data=True)
            ],
        }

    def total_tokens(self) -> int:
        total = 0
        for _, data in self.G.nodes(data=True):
            total += (data.get("prompt_tok", 0) or 0) + (data.get("compl_tok", 0) or 0)
        return total
