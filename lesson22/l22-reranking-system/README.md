# L22: Retrieval Optimization - Reranking System

Production-grade reranking service with cross-encoder models for RAG optimization.

## Quick Start

```bash
# Build
./build.sh

# Start services
./start.sh

# Test
./test.sh

# Stop
./stop.sh
```

## Access Points

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- Cross-encoder reranking with ms-marco-MiniLM
- Redis caching (24hr TTL)
- Real-time metrics dashboard
- Bi-encoder vs cross-encoder comparison
- Batch reranking support
- Production-grade error handling

## Architecture

- FastAPI backend with Sentence Transformers
- React dashboard with Recharts
- Redis for result caching
- Docker Compose orchestration

## Integration with L21

Builds on L21's embedding and chunking infrastructure, adding post-retrieval
optimization layer for improved relevance.

## Preparing for L23

Reranking provides unified scoring for hybrid search combining vector and
keyword search results.
