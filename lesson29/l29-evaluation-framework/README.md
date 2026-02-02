# L29: Evaluation Framework

Tool & RAG accuracy and latency evaluation for VAIA agents.

## Quick Start

```bash
# From lesson29/
./setup.sh

# Or from project root:
cd l29-evaluation-framework
./scripts/build.sh
./scripts/start.sh
```

## Run Demo

1. Start services: `./scripts/start.sh`
2. Open http://localhost:3000
3. Click "Run Full Evaluation" to populate dashboard metrics
4. Or run: `./scripts/demo.sh` (requires backend running)

## Endpoints

- Dashboard: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Metrics

Dashboard displays:
- Overall Accuracy, Tool Call Accuracy, RAG Groundedness
- P99 Latency, Tests Run
- Values update after each evaluation run

Set `GEMINI_API_KEY` in `backend/.env` for real LLM evaluation.
