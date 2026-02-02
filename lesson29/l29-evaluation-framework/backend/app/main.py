from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from app.services.evaluation_service import EvaluationService
from app.services.metrics_service import MetricsService
from app.models.schemas import (
    EvaluationRequest, EvaluationResponse,
    TestSuiteRequest, MetricsResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

evaluation_service = None
metrics_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluation_service, metrics_service
    logger.info("Initializing evaluation services...")
    evaluation_service = EvaluationService()
    metrics_service = MetricsService()
    await evaluation_service.initialize()
    await metrics_service.initialize()
    yield
    logger.info("Shutting down services...")
    await evaluation_service.shutdown()

app = FastAPI(
    title="VAIA Evaluation Framework",
    description="L29: Tool & RAG Accuracy/Latency Evaluation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "service": "evaluation-framework"}

@app.post("/api/evaluate/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    try:
        logger.info(f"Running evaluation: {request.test_suite_name}")
        result = await evaluation_service.run_evaluation(
            test_suite_name=request.test_suite_name,
            iterations=request.iterations
        )
        metrics_service.record_evaluation(result.model_dump())
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/tool-calls")
async def evaluate_tool_calls(request: TestSuiteRequest):
    try:
        return await evaluation_service.evaluate_tool_calls(test_cases=request.test_cases)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/rag-groundedness")
async def evaluate_rag_groundedness(request: TestSuiteRequest):
    try:
        return await evaluation_service.evaluate_rag_groundedness(test_cases=request.test_cases)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/latency")
async def evaluate_latency(request: EvaluationRequest):
    try:
        return await evaluation_service.evaluate_latency(
            test_suite_name=request.test_suite_name,
            iterations=request.iterations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary():
    try:
        return await metrics_service.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/history")
async def get_metrics_history(hours: int = 24):
    try:
        return await metrics_service.get_history(hours=hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-suites")
async def list_test_suites():
    return await evaluation_service.list_test_suites()

@app.post("/api/regression/detect")
async def detect_regressions():
    try:
        return await metrics_service.detect_regressions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
