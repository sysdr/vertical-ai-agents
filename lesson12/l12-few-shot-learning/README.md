# L12: Few-Shot Prompt Engineering System

Enterprise-grade few-shot learning system for classification tasks.

## Features
- Dynamic example selection with similarity search
- Performance benchmarking across shot counts (0, 1, 3, 5)
- Example management and storage
- Real-time metrics dashboard
- Token counting and latency tracking

## Quick Start

### Standard Setup
```bash
chmod +x build.sh start.sh stop.sh test.sh
./build.sh
./start.sh
```

### Docker Setup
```bash
docker-compose up -d
```

## Access
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Usage
1. Navigate to **Classify** tab to test classifications
2. Use **Examples** tab to manage training examples
3. Run **Benchmark** to compare performance across shot counts
4. View **Metrics** for performance insights

## Testing
```bash
./test.sh
```

## Architecture
- Backend: Python FastAPI with Gemini AI
- Frontend: React with real-time updates
- Storage: JSON-based example store with embeddings
- Similarity search: Cosine similarity with semantic embeddings
