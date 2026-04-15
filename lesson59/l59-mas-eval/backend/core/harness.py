"""
EvaluationHarness: correctness, efficiency, and topology scoring for MAS runs.
"""
from __future__ import annotations
import re
from typing import Optional

from models.trace import TestCase, TestResult, EvalReport
from core.graph import InteractionGraph
from core.session import DebugSession


class EvaluationHarness:
    def __init__(self):
        self._test_cases: list[TestCase] = []

    def add_test(self, tc: TestCase) -> None:
        self._test_cases.append(tc)

    def score(self, session: DebugSession) -> EvalReport:
        graph = InteractionGraph(session.records)
        results: list[TestResult] = []

        for tc in self._test_cases:
            correctness = self._check_output(session.final_output, tc)
            efficiency  = self._check_efficiency(graph, session.records, tc)
            results.append(TestResult(
                test_name=tc.name,
                correctness=correctness,
                efficiency=efficiency,
                details={
                    "output_len": len(session.final_output),
                    "total_tokens": graph.total_tokens(),
                },
            ))

        passed = sum(1 for r in results if r.correctness and r.efficiency)
        return EvalReport(
            test_results=results,
            agent_hotspots=graph.agent_hotspots(),
            critical_path=graph.critical_path(),
            total_tokens=graph.total_tokens(),
            passed=passed,
            failed=len(results) - passed,
        )

    def _check_output(self, output: str, tc: TestCase) -> bool:
        if not tc.expected_output_pattern:
            return bool(output.strip())
        return bool(re.search(tc.expected_output_pattern, output, re.IGNORECASE | re.DOTALL))

    def _check_efficiency(
        self, graph: InteractionGraph, records: list, tc: TestCase
    ) -> bool:
        total_tokens = graph.total_tokens()
        if total_tokens > tc.max_tokens:
            return False
        if records:
            from models.trace import TraceRecord
            first = records[0].timestamp if hasattr(records[0], "timestamp") else 0
            last  = records[-1].timestamp if hasattr(records[-1], "timestamp") else 0
            wall_ms = (last - first) * 1000
            if wall_ms > tc.max_latency_ms:
                return False
        return True
