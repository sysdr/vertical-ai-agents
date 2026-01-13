# L20: Vector Databases & Indexing

## Quick Start

### 1. Set up API Key
**Option A: Use the helper script (recommended)**
```bash
./set-api-key.sh YOUR_API_KEY_HERE
```

**Option B: Manual setup**
Create a `.env` file in the project root:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
docker-compose restart backend
```

Get your API key from: https://makersuite.google.com/app/apikey

### 2. Build and Start
```bash
./build.sh   # Install dependencies
./start.sh   # Start services
./test.sh    # Run tests
```

Dashboard: http://localhost:3000
API: http://localhost:8000/docs

**Note**: If you see "API key expired or invalid" errors, update the `GEMINI_API_KEY` in your `.env` file and restart the services.

## Architecture

- **Backend**: FastAPI + ChromaDB + Gemini Embeddings
- **Frontend**: React + Real-time WebSocket updates
- **Storage**: Persistent ChromaDB with HNSW indexing

## Features

- Character-based document chunking
- Batch embedding with Gemini API
- Semantic search with metadata filtering
- Real-time collection statistics
- Production-ready error handling
