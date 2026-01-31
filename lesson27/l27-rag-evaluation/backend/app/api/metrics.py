from fastapi import APIRouter
from ..schemas.evaluation import MetricsSummary
from .evaluation import evaluation_runs

router = APIRouter()

# Baseline when no runs exist (non-zero demo values)
BASELINE = {
    "avg_faithfulness": 0.87,
    "avg_relevance": 0.82,
    "avg_precision": 0.79,
    "avg_recall": 0.75,
}

def _compute_metrics_from_runs():
    """Use latest run's results for real-time metric display"""
    completed = [r for r in evaluation_runs.values() if r.get("status") == "completed" and r.get("results")]
    if not completed:
        return None
    # Sort by run_id desc, use most recent run's results (real-time feel)
    latest = max(completed, key=lambda r: r.get("run_id", 0))
    res = latest["results"]
    return {
        "avg_faithfulness": round(float(res["faithfulness"]), 2),
        "avg_relevance": round(float(res["answer_relevance"]), 2),
        "avg_precision": round(float(res["context_precision"]), 2),
        "avg_recall": round(float(res["context_recall"]), 2),
        "total_evaluations": sum(r.get("total_queries", 0) for r in completed),
        "period": "last_24h",
    }

@router.get("/summary", response_model=MetricsSummary)
async def get_metrics_summary():
    """Get aggregated metrics from evaluation runs, or baseline"""
    metrics = _compute_metrics_from_runs()
    if metrics:
        return MetricsSummary(**metrics)
    return MetricsSummary(
        avg_faithfulness=BASELINE["avg_faithfulness"],
        avg_relevance=BASELINE["avg_relevance"],
        avg_precision=BASELINE["avg_precision"],
        avg_recall=BASELINE["avg_recall"],
        total_evaluations=0,
        period="last_24h"
    )

@router.get("/trends")
async def get_trends():
    """Get metric trends - uses run data when available"""
    completed = [r for r in evaluation_runs.values() if r.get("status") == "completed" and r.get("results")]
    trends = []
    if completed:
        for i, r in enumerate(completed[-24:]):
            res = r["results"]
            trends.append({
                "timestamp": r.get("run_name", f"Run {i+1}")[:12],
                "faithfulness": round(res["faithfulness"], 2),
                "relevance": round(res["answer_relevance"], 2),
                "precision": round(res["context_precision"], 2),
                "recall": round(res["context_recall"], 2)
            })
    if not trends:
        for i in range(24):
            trends.append({
                "timestamp": f"{i}:00",
                "faithfulness": BASELINE["avg_faithfulness"],
                "relevance": BASELINE["avg_relevance"],
                "precision": BASELINE["avg_precision"],
                "recall": BASELINE["avg_recall"]
            })
    return {"trends": trends}
