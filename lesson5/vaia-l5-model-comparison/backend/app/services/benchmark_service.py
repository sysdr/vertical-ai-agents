import asyncio
import statistics
from typing import List, Dict, Any

class BenchmarkService:
    """Service for orchestrating model benchmarks."""
    
    def __init__(self, model_client, analytics_engine):
        self.model_client = model_client
        self.analytics_engine = analytics_engine
        
    async def run_benchmark(
        self,
        prompts: List[str],
        models: List[str],
        repetitions: int = 3,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Execute benchmark across models and prompts."""
        
        results = []
        
        for model in models:
            model_results = {
                "model": model,
                "prompts": []
            }
            
            for prompt in prompts:
                prompt_results = []
                
                # Run multiple repetitions
                for _ in range(repetitions):
                    result = await self.model_client.generate(
                        model=model,
                        prompt=prompt,
                        temperature=temperature
                    )
                    
                    if result["success"]:
                        cost = self.model_client.calculate_cost(
                            model=model,
                            input_tokens=result["input_tokens"],
                            output_tokens=result["output_tokens"]
                        )
                        
                        prompt_results.append({
                            "latency_ms": result["latency_ms"],
                            "cost": cost,
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "response_length": len(result["text"])
                        })
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                
                # Aggregate statistics
                if prompt_results:
                    latencies = [r["latency_ms"] for r in prompt_results]
                    costs = [r["cost"] for r in prompt_results]
                    
                    model_results["prompts"].append({
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "avg_latency_ms": statistics.mean(latencies),
                        "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                        "avg_cost": statistics.mean(costs),
                        "total_runs": len(prompt_results),
                        "success_rate": 1.0
                    })
            
            results.append(model_results)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return {
            "detailed_results": results,
            "summary": summary,
            "metadata": {
                "total_prompts": len(prompts),
                "total_models": len(models),
                "repetitions": repetitions,
                "temperature": temperature
            }
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics across all results."""
        
        summary = {}
        
        for model_result in results:
            model_name = model_result["model"]
            
            all_latencies = []
            all_costs = []
            
            for prompt_result in model_result["prompts"]:
                all_latencies.append(prompt_result["avg_latency_ms"])
                all_costs.append(prompt_result["avg_cost"])
            
            if all_latencies and all_costs:
                summary[model_name] = {
                    "avg_latency_ms": statistics.mean(all_latencies),
                    "avg_cost_per_request": statistics.mean(all_costs),
                    "total_requests": len(all_latencies)
                }
        
        return summary
