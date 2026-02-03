from fastapi import APIRouter, HTTPException
from app.models.schemas import PlanningRequest, PlanningResponse
from app.services.react_planner import ReActPlanner
from app.services.metrics_collector import metrics

router = APIRouter()
planner = ReActPlanner()

@router.post("/plan", response_model=PlanningResponse)
async def create_plan(request: PlanningRequest):
    """Execute ReAct planning for a task"""
    try:
        trace = await planner.plan(request)
        avg_conf = sum(s.reasoning.confidence for s in trace.steps) / len(trace.steps) if trace.steps else 0
        metrics.record_plan(
            iterations=trace.total_iterations,
            tokens=trace.total_tokens,
            latency_ms=trace.total_latency_ms,
            avg_confidence=avg_conf
        )
        return PlanningResponse(task_id=trace.task_id, trace=trace)
    except ValueError as e:
        if "GEMINI_API_KEY" in str(e) or "API" in str(e):
            raise HTTPException(
                status_code=400,
                detail="API key not configured or invalid. Get a free key at https://aistudio.google.com/apikey and set GEMINI_API_KEY in .env. See API_KEY_SETUP.md"
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        err = str(e)
        if "API key expired" in err or "API_KEY_INVALID" in err or "400" in err:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key expired or invalid. Get a new key at https://aistudio.google.com/apikey and update GEMINI_API_KEY in .env"
            )
        raise HTTPException(status_code=500, detail=err)

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "react-planner"}

@router.get("/stats")
async def get_stats():
    """Get aggregated planning metrics for dashboard"""
    return metrics.get_stats()
