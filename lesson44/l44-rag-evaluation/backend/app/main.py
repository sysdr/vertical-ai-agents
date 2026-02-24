"""
L44 FastAPI Server — Evaluation pipeline endpoints.

Endpoints:
  POST /api/evaluate          — Start evaluation run
  GET  /api/evaluations/{id}  — Fetch specific report
  GET  /api/evaluations/latest — Latest report
  GET  /api/evaluations       — List all reports
  GET  /api/metrics/thresholds — Get deployment thresholds
  GET  /health                — Health check
"""
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import EvaluationRequest, EvaluationReport, EvalState, DEPLOYMENT_THRESHOLDS
from app.evaluator import EvaluationOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "data/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# In-memory state for active runs
active_runs: dict[str, EvaluationReport] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("L44 RAG Evaluation Server starting...")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — evaluation will fail")
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="L44 · Agentic RAG Evaluation API",
    description="Evaluation pipeline for Agentic RAG systems — VAIA Curriculum L44",
    version="44.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_evaluation_task(report_id: str, request: EvaluationRequest):
    """Background task: run the full evaluation pipeline."""
    orchestrator = EvaluationOrchestrator(api_key=GEMINI_API_KEY)
    report = await orchestrator.run_evaluation(
        run_name=request.run_name,
        num_questions=request.num_test_questions,
        use_synthetic=request.use_synthetic_dataset,
    )
    report.id = report_id
    active_runs[report_id] = report
    # Persist
    path = REPORTS_DIR / f"{report_id}.json"
    path.write_text(report.model_dump_json(indent=2))
    (REPORTS_DIR / "latest.json").write_text(report.model_dump_json(indent=2))


@app.post("/api/evaluate", response_model=dict)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    import uuid
    report_id = str(uuid.uuid4())
    # Create placeholder in active_runs
    active_runs[report_id] = EvaluationReport(
        id=report_id,
        run_name=request.run_name,
        state=EvalState.DATASET_LOADING,
    )
    background_tasks.add_task(run_evaluation_task, report_id, request)
    return {"report_id": report_id, "status": "started", "run_name": request.run_name}


@app.get("/api/evaluations/latest")
async def get_latest_report():
    # Check active_runs for most recent
    if active_runs:
        latest = max(active_runs.values(), key=lambda r: r.started_at or r.id)
        return latest.model_dump()
    # Check persisted
    latest_path = REPORTS_DIR / "latest.json"
    if latest_path.exists():
        import json
        return json.loads(latest_path.read_text())
    raise HTTPException(status_code=404, detail="No evaluations found yet")


@app.get("/api/evaluations/{report_id}")
async def get_report(report_id: str):
    # Check active runs first
    if report_id in active_runs:
        return active_runs[report_id].model_dump()
    # Check persisted
    path = REPORTS_DIR / f"{report_id}.json"
    if path.exists():
        import json
        return json.loads(path.read_text())
    raise HTTPException(status_code=404, detail=f"Report {report_id} not found")


@app.get("/api/evaluations")
async def list_reports():
    import json
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        if p.name == "latest.json":
            continue
        try:
            data = json.loads(p.read_text())
            reports.append({
                "id": data.get("id"),
                "run_name": data.get("run_name"),
                "state": data.get("state"),
                "gate_passed": data.get("gate_passed"),
                "total_questions": data.get("total_questions"),
                "aggregate": data.get("aggregate"),
                "started_at": data.get("started_at"),
            })
        except Exception:
            continue
    # Merge with active in-memory runs
    for r in active_runs.values():
        if not any(rep["id"] == r.id for rep in reports):
            reports.insert(0, {
                "id": r.id,
                "run_name": r.run_name,
                "state": r.state,
                "gate_passed": r.gate_passed,
                "total_questions": r.total_questions,
                "aggregate": r.aggregate.model_dump() if r.aggregate else None,
                "started_at": str(r.started_at),
            })
    return reports[:20]


@app.get("/api/metrics/thresholds")
async def get_thresholds():
    return DEPLOYMENT_THRESHOLDS


@app.get("/health")
async def health():
    return {"status": "ok", "service": "L44 RAG Evaluator", "gemini_configured": bool(GEMINI_API_KEY)}
