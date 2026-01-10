# Lesson 19: Naïve RAG System

A production-ready Retrieval Augmented Generation system with document chunking, in-memory knowledge base, and query-retrieve-respond pipeline.

## Quick Start

```bash
# Build the system
./build.sh

# Start services
./start.sh

# Access the dashboard
open http://localhost:3000

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Features

- Document ingestion with character-based chunking (500-char chunks, 50-char overlap)
- In-memory knowledge base with keyword indexing
- Fast retrieval (<50ms for queries)
- Gemini AI integration for answer generation
- Real-time dashboard with metrics
- Complete test suite

## Architecture

- Backend: FastAPI (Python 3.11)
- Frontend: React 18
- LLM: Google Gemini 2.0 Flash
- Deployment: Docker + local options

## API Endpoints

- `POST /documents/upload` - Upload and chunk document
- `POST /query` - Query knowledge base
- `GET /stats` - System statistics
- `GET /health` - Health check

## Performance

- Ingestion: ~1,000 chunks/second
- Retrieval: <50ms average
- End-to-end query: <500ms

## Next Steps

Lesson 20 will replace in-memory storage with ChromaDB for vector-based semantic search, improving retrieval quality for larger knowledge bases.
