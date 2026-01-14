# L21: Advanced Embeddings & Chunking

Production-grade RAG preprocessing laboratory for VAIA systems.

## Quick Start

```bash
# Build project
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Docker

```bash
docker-compose up --build
```

## Features

- 5 chunking strategies (fixed, semantic, recursive, sliding_window, sentence)
- 3 embedding models (Sentence Transformers, Gemini, lightweight)
- Real-time comparison dashboard
- ChromaDB integration (builds on L20)
- Quality metrics and visualization

## API Endpoints

- POST /chunk - Apply chunking strategy
- POST /embed - Generate embeddings
- POST /compare - Compare strategies
- POST /upload - Upload document
- GET /strategies - List available options
- GET /health - Health check

## Architecture

- Backend: Python FastAPI
- Frontend: React
- Vector DB: ChromaDB
- AI: Gemini + Sentence Transformers

Access dashboard: http://localhost:3000
