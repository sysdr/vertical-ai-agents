import time
import asyncio
from typing import Dict

class LatencyProfiler:
    async def profile_query(self, query: str, agent_service) -> Dict[str, float]:
        latencies = {}
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        await asyncio.sleep(0.02)
        latencies['retrieval'] = (time.perf_counter() - retrieval_start) * 1000
        llm_start = time.perf_counter()
        response = await agent_service.process_query(query)
        latencies['llm_inference'] = (time.perf_counter() - llm_start) * 1000
        if response.get('tool_calls'):
            tool_start = time.perf_counter()
            await asyncio.sleep(0.01)
            latencies['tool_execution'] = (time.perf_counter() - tool_start) * 1000
        latencies['total'] = (time.perf_counter() - total_start) * 1000
        return latencies
