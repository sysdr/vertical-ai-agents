# L44 · Evaluating Agentic RAG Reliability

VAIA Curriculum — Module 4 — Lesson 44

## Quick Start

```bash
# 1. Build (creates venv, installs deps)
./build.sh

# 2. Start services
./start.sh

# 3. Open dashboard
open http://localhost:3000

# 4. Run tests
./test.sh
```

## Docker

```bash
docker-compose up --build
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/evaluate` | POST | Start evaluation run |
| `/api/evaluations/latest` | GET | Latest report |
| `/api/evaluations/{id}` | GET | Specific report |
| `/api/evaluations` | GET | List all reports |
| `/api/metrics/thresholds` | GET | Deployment thresholds |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |

## Metrics & Thresholds

| Metric | Threshold | What it measures |
|---|---|---|
| Faithfulness | ≥ 85% | Anti-hallucination |
| Answer Relevancy | ≥ 80% | On-topic responses |
| Context Recall | ≥ 75% | Retrieval completeness |
| Context Precision | ≥ 70% | Retrieval precision |

## Connects to L43

L44 consumes `TraceRecord` objects from L43's Agentic RAG pipeline.
Point `CHROMA_HOST` to L43's ChromaDB for live evaluation.

## Enables L45

The `EvaluationOrchestrator` and `MetricScores` models are imported
directly by L45's Autonomous Research Agent as quality gates.
