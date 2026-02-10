from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and aggregate retrieval metrics"""
    
    def __init__(self):
        self.retrievals: List[Dict] = []
        self.start_time = datetime.now()
    
    def record_retrieval(self, metrics: Dict):
        """Record retrieval metrics"""
        self.retrievals.append({
            **metrics,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.retrievals:
            return {"message": "No retrievals yet"}
        
        total = len(self.retrievals)
        avg_time = sum(r["execution_time_ms"] for r in self.retrievals) / total
        
        reranked = [r for r in self.retrievals if r.get("reranking_enabled")]
        hybrid = [r for r in self.retrievals if r.get("hybrid_enabled")]
        
        return {
            "total_retrievals": total,
            "avg_execution_time_ms": round(avg_time, 2),
            "reranking_usage_pct": round(len(reranked) / total * 100, 1),
            "hybrid_usage_pct": round(len(hybrid) / total * 100, 1),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "recent_retrievals": self.retrievals[-10:]
        }
