"""In-memory metrics collector for planning observability"""
from collections import defaultdict
from typing import Dict, List
from datetime import datetime

class MetricsCollector:
    def __init__(self):
        self.total_plans = 0
        self.total_iterations = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.latency_samples: List[float] = []
        self.confidence_samples: List[float] = []
        
    def record_plan(self, iterations: int, tokens: int, latency_ms: float, avg_confidence: float = 0):
        self.total_plans += 1
        self.total_iterations += iterations
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.latency_samples.append(latency_ms)
        if avg_confidence > 0:
            self.confidence_samples.append(avg_confidence)
            
    def get_stats(self) -> Dict:
        avg_latency = self.total_latency_ms / self.total_plans if self.total_plans else 0
        avg_confidence = sum(self.confidence_samples) / len(self.confidence_samples) if self.confidence_samples else 0
        return {
            "total_plans": self.total_plans,
            "total_iterations": self.total_iterations,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_confidence": round(avg_confidence, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }

metrics = MetricsCollector()
