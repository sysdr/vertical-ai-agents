import asyncio
import time
import logging
import json
import os
from typing import List
from datetime import datetime

from app.models.schemas import (
    TestCase, ToolCallResult, RAGGroundednessResult,
    LatencyResult, EvaluationResponse
)
from app.services.agent_service import AgentService
from app.services.validators import ToolCallValidator, RAGGroundednessValidator
from app.services.latency_profiler import LatencyProfiler

logger = logging.getLogger(__name__)

def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

class EvaluationService:
    def __init__(self):
        self.agent_service = None
        self.tool_validator = ToolCallValidator()
        self.rag_validator = RAGGroundednessValidator()
        self.latency_profiler = LatencyProfiler()

    async def initialize(self):
        self.agent_service = AgentService()
        await self.agent_service.initialize()
        logger.info("Evaluation service initialized")

    async def shutdown(self):
        if self.agent_service:
            await self.agent_service.shutdown()

    async def run_evaluation(self, test_suite_name: str, iterations: int = 3) -> EvaluationResponse:
        start_time = time.time()
        test_cases = await self._load_test_suite(test_suite_name)
        iterations = min(iterations, 5)
        tool_results, rag_results, latency_results = await asyncio.gather(
            self.evaluate_tool_calls(test_cases),
            self.evaluate_rag_groundedness(test_cases),
            self._evaluate_latency_detailed(test_cases, iterations)
        )
        total = len(test_cases) * iterations
        passed = sum(1 for r in tool_results if r.success) + sum(1 for r in rag_results if r.success)
        tool_acc = sum(1 for r in tool_results if r.success) / len(tool_results) if tool_results else 0
        rag_acc = sum(r.groundedness_score for r in rag_results) / len(rag_results) if rag_results else 0
        resp = EvaluationResponse(
            test_suite=test_suite_name,
            total_tests=total,
            passed=passed,
            failed=total - passed,
            tool_call_accuracy=tool_acc,
            rag_groundedness=rag_acc,
            latency_summary=latency_results,
            timestamp=datetime.utcnow(),
            duration_seconds=time.time() - start_time
        )
        return resp

    async def evaluate_tool_calls(self, test_cases: List[TestCase]) -> List[ToolCallResult]:
        results = []
        for tc in test_cases:
            if not tc.expected_tool:
                continue
            start = time.perf_counter()
            response = await self.agent_service.process_query(tc.query)
            latency_ms = (time.perf_counter() - start) * 1000
            v = await self.tool_validator.validate(response, tc.expected_tool, tc.expected_params or {})
            results.append(ToolCallResult(
                test_case_id=tc.id,
                actual_tool=v.get('actual_tool'),
                expected_tool=tc.expected_tool,
                parameters_match=v.get('param_score', 0),
                success=v.get('success', False),
                latency_ms=latency_ms
            ))
        return results

    async def evaluate_rag_groundedness(self, test_cases: List[TestCase]) -> List[RAGGroundednessResult]:
        results = []
        for tc in test_cases:
            if not tc.ground_truth_docs:
                continue
            response = await self.agent_service.process_query(tc.query)
            v = await self.rag_validator.validate(
                response.get('answer', ''),
                response.get('retrieved_chunks', []),
                tc.ground_truth_docs
            )
            results.append(RAGGroundednessResult(
                test_case_id=tc.id,
                response=response.get('answer', ''),
                groundedness_score=v.get('score', 0),
                unsupported_claims=v.get('unsupported_claims', []),
                retrieved_chunks=len(response.get('retrieved_chunks', [])),
                success=v.get('score', 0) >= 0.7
            ))
        return results

    async def evaluate_latency(self, test_suite_name: str, iterations: int = 3) -> List[LatencyResult]:
        test_cases = await self._load_test_suite(test_suite_name)
        return await self._evaluate_latency_detailed(test_cases, min(iterations, 5))

    async def _evaluate_latency_detailed(self, test_cases: List[TestCase], iterations: int) -> List[LatencyResult]:
        components = ['retrieval', 'tool_execution', 'llm_inference', 'total']
        latencies = {c: [] for c in components}
        for _ in range(iterations):
            for tc in test_cases[:5]:
                profile = await self.latency_profiler.profile_query(tc.query, self.agent_service)
                for c in components:
                    if c in profile:
                        latencies[c].append(profile[c])
        results = []
        for comp, times in latencies.items():
            if not times:
                continue
            st = sorted(times)
            n = len(st)
            results.append(LatencyResult(
                component=comp,
                mean_ms=sum(times) / n,
                p50_ms=st[n // 2],
                p95_ms=st[min(int(n * 0.95), n - 1)],
                p99_ms=st[min(int(n * 0.99), n - 1)],
                max_ms=max(times),
                samples=n
            ))
        return results

    async def _load_test_suite(self, name: str) -> List[TestCase]:
        root = _project_root()
        path = os.path.join(root, 'data', 'test_suites', f'{name}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                return [TestCase(**c) for c in data.get('test_cases', [])]
        return self._get_default_test_suite()

    def _get_default_test_suite(self) -> List[TestCase]:
        return [
            TestCase(id="tc001", type="tool_call", query="What's the weather in San Francisco?",
                     expected_tool="get_weather", expected_params={"location": "San Francisco"}),
            TestCase(id="tc002", type="rag_groundedness", query="What are the main features of Python?",
                     ground_truth_docs=["Python is an interpreted, high-level programming language."]),
            TestCase(id="tc003", type="tool_call", query="Search for restaurants near me",
                     expected_tool="search_places", expected_params={"query": "restaurants", "nearby": True})
        ]

    async def list_test_suites(self) -> List[str]:
        root = _project_root()
        d = os.path.join(root, 'data', 'test_suites')
        if os.path.exists(d):
            return [f.replace('.json', '') for f in os.listdir(d) if f.endswith('.json')]
        return ["default"]
