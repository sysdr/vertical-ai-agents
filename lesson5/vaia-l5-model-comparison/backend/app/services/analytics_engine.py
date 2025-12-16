from typing import List, Dict, Optional

class AnalyticsEngine:
    """Engine for analyzing benchmark results and generating insights."""
    
    def __init__(self):
        self.benchmark_data = {}
        
    async def compare_models(self, model_names: List[str]) -> Dict:
        """Compare models across multiple dimensions."""
        
        # Simulated comparison data (in production, this would use real benchmark results)
        comparisons = {
            "performance": {
                "gemini-2.0-flash": {"latency": 75, "throughput": 45},
                "gemini-1.5-pro": {"latency": 180, "throughput": 20},
                "gemini-1.5-flash": {"latency": 60, "throughput": 50}
            },
            "cost": {
                "gemini-2.0-flash": {"per_1k_tokens": 0.001, "per_request": 0.0015},
                "gemini-1.5-pro": {"per_1k_tokens": 0.00425, "per_request": 0.006},
                "gemini-1.5-flash": {"per_1k_tokens": 0.0002, "per_request": 0.0003}
            },
            "quality": {
                "gemini-2.0-flash": {"mmlu_score": 82.5, "reasoning": "good"},
                "gemini-1.5-pro": {"mmlu_score": 88.9, "reasoning": "excellent"},
                "gemini-1.5-flash": {"mmlu_score": 78.9, "reasoning": "good"}
            }
        }
        
        result = {
            "models": model_names,
            "comparisons": {}
        }
        
        for model in model_names:
            if model in comparisons["performance"]:
                result["comparisons"][model] = {
                    "performance": comparisons["performance"][model],
                    "cost": comparisons["cost"][model],
                    "quality": comparisons["quality"][model]
                }
        
        # Calculate Pareto frontier
        result["pareto_frontier"] = self._calculate_pareto_frontier(
            result["comparisons"]
        )
        
        return result
    
    def _calculate_pareto_frontier(self, comparisons: Dict) -> List[str]:
        """Identify models on the Pareto frontier."""
        
        frontier = []
        
        for model, data in comparisons.items():
            dominated = False
            
            for other_model, other_data in comparisons.items():
                if model == other_model:
                    continue
                
                # Check if other model dominates this one
                better_latency = other_data["performance"]["latency"] <= data["performance"]["latency"]
                better_cost = other_data["cost"]["per_request"] <= data["cost"]["per_request"]
                better_quality = other_data["quality"]["mmlu_score"] >= data["quality"]["mmlu_score"]
                
                if better_latency and better_cost and better_quality:
                    if (other_data["performance"]["latency"] < data["performance"]["latency"] or
                        other_data["cost"]["per_request"] < data["cost"]["per_request"] or
                        other_data["quality"]["mmlu_score"] > data["quality"]["mmlu_score"]):
                        dominated = True
                        break
            
            if not dominated:
                frontier.append(model)
        
        return frontier
    
    async def generate_recommendations(
        self,
        max_latency_ms: Optional[int] = None,
        max_cost_per_request: Optional[float] = None,
        min_quality_score: Optional[float] = None
    ) -> Dict:
        """Generate model recommendations based on constraints."""
        
        # Model specifications
        models = {
            "gemini-2.0-flash": {
                "latency_ms": 75,
                "cost_per_request": 0.0015,
                "quality_score": 82.5,
                "use_case": "High-throughput applications"
            },
            "gemini-1.5-pro": {
                "latency_ms": 180,
                "cost_per_request": 0.006,
                "quality_score": 88.9,
                "use_case": "Complex reasoning tasks"
            },
            "gemini-1.5-flash": {
                "latency_ms": 60,
                "cost_per_request": 0.0003,
                "quality_score": 78.9,
                "use_case": "Cost-sensitive, high-volume scenarios"
            }
        }
        
        recommendations = []
        
        for model_name, specs in models.items():
            meets_constraints = True
            
            if max_latency_ms and specs["latency_ms"] > max_latency_ms:
                meets_constraints = False
            
            if max_cost_per_request and specs["cost_per_request"] > max_cost_per_request:
                meets_constraints = False
            
            if min_quality_score and specs["quality_score"] < min_quality_score:
                meets_constraints = False
            
            if meets_constraints:
                recommendations.append({
                    "model": model_name,
                    "specs": specs,
                    "confidence": "high"
                })
        
        # Sort by cost efficiency
        recommendations.sort(key=lambda x: x["specs"]["cost_per_request"])
        
        return {
            "recommended_models": recommendations,
            "constraints": {
                "max_latency_ms": max_latency_ms,
                "max_cost_per_request": max_cost_per_request,
                "min_quality_score": min_quality_score
            },
            "routing_strategy": self._suggest_routing_strategy(recommendations)
        }
    
    def _suggest_routing_strategy(self, recommendations: List[Dict]) -> Dict:
        """Suggest routing strategy based on available models."""
        
        if len(recommendations) >= 2:
            return {
                "strategy": "cascading",
                "description": "Route simple queries to efficient models, escalate complex ones",
                "primary_model": recommendations[0]["model"],
                "fallback_model": recommendations[-1]["model"]
            }
        elif len(recommendations) == 1:
            return {
                "strategy": "single_model",
                "description": "Use single model for all queries",
                "primary_model": recommendations[0]["model"]
            }
        else:
            return {
                "strategy": "none",
                "description": "No models meet constraints"
            }
