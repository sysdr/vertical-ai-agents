# L24: Modular RAG with LangChain

Enterprise-grade RAG system using LangChain for flexible, composable retrieval chains.

## Quick Start

### Docker (Recommended)
```bash
./scripts/build.sh
./scripts/start.sh
./scripts/test.sh
```

### Local Development
```bash
./scripts/setup-local.sh
source venv/bin/activate
python backend/main.py  # Terminal 1
cd frontend && npm start  # Terminal 2
```

## Features

- **Hybrid Retrieval**: Combines vector and keyword search in LangChain retrievers
- **LCEL Chains**: Declarative chain composition using LangChain Expression Language
- **Modular Architecture**: Swap retrievers, prompts, and LLMs without code changes
- **Production Ready**: Metrics, logging, error handling, and observability
- **Gemini Integration**: Google's latest Gemini 2.0 Flash model

## Architecture

- Backend: FastAPI + LangChain + ChromaDB
- Frontend: React with real-time query interface
- LLM: Gemini 2.0 Flash Exp
- Embeddings: Google Embedding-001

## API Endpoints

- `POST /api/query` - RAG query with chain execution
- `POST /api/documents` - Add documents to vectorstore
- `GET /api/stats` - System statistics
- `GET /metrics` - Prometheus metrics

## URLs

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics
