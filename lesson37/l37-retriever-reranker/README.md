# L37: RetrieverAgent & Reranker Tool

## Overview
Multi-stage retrieval system with query decomposition, parallel search, and semantic reranking.

## Architecture
- **PlannerAgent**: Query decomposition (from L36)
- **RetrieverAgent**: Orchestrates search pipeline
- **RerankerTool**: Cross-encoder semantic scoring
- **VectorStore**: ChromaDB + BM25 hybrid search

## Quick Start

### Docker (recommended)
```bash
export GEMINI_API_KEY=your-key-here
docker compose up --build
```

### Stop and cleanup
From this directory: `./cleanup.sh`

## Features
- ✅ Multi-stage retrieval: plan → search → rerank
- ✅ Hybrid search (vector + BM25 + RRF)
- ✅ Parallel sub-query execution
- ✅ Real-time WebSocket updates
- ✅ Metrics dashboard
- ✅ Graceful fallback on reranker failures

## URLs
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Lesson Context
- **Previous (L36)**: PlannerAgent query decomposition
- **Current (L37)**: RetrieverAgent with reranking
- **Next (L38)**: ValidatorAgent factual consistency
