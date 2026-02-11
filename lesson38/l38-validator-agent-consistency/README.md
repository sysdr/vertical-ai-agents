# L38: The Validator Agent - Factual Consistency

## Overview

ValidatorAgent checks retrieved documents for factual consistency using Gemini LLM. Detects contradictions, quantifies agreement, and provides validation reports before generation.

## Features

- **Pairwise Consistency Checking**: Compare all document pairs for contradictions
- **LLM-Based Analysis**: Use Gemini to understand semantic conflicts
- **Redis Caching**: Cache validation results with 1-hour TTL
- **Parallel Processing**: Validate multiple pairs concurrently
- **Real-Time Dashboard**: See validation results as they complete
- **Production-Ready**: Timeout protection, graceful degradation, comprehensive error handling

## Architecture
```
Request → ValidatorAgent
  ↓
Generate document pairs
  ↓
For each pair:
  Check cache → HIT: return cached
             → MISS: call Gemini → cache result
  ↓
Aggregate scores
  ↓
Flag contradictions (score < 0.5)
  ↓
Return validation report
```

## Quick Start

### Option 1: Local Setup
```bash
# Run automated setup
./setup.sh

# Build project
./scripts/build.sh

# Start services
./scripts/start.sh

# Access dashboard
open http://localhost:3000

# Stop services
./scripts/stop.sh
```

### Option 2: Docker
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Usage

1. **Add Documents**: Enter or paste document content in the dashboard
2. **Set Metadata**: Specify source and relevance score for each document
3. **Validate**: Click "Validate Consistency" to check for contradictions
4. **Review Results**: See overall consistency, pairwise scores, and flagged conflicts

## API Endpoints
```bash
# Validate documents
POST /api/v1/validator/validate
Body: [
  {
    "id": "doc1",
    "content": "Document text...",
    "score": 0.95,
    "source": "Financial Report",
    "metadata": {}
  }
]

# Health check
GET /api/v1/validator/health
```

## Configuration

Edit `backend/config/settings.py`:
```python
MAX_CONCURRENT_VALIDATIONS = 5  # Parallel Gemini calls
VALIDATION_TIMEOUT = 10  # Seconds per validation
CONSISTENCY_THRESHOLD = 0.5  # Flag conflicts below this
CACHE_TTL = 3600  # Redis cache duration (seconds)
```

## Testing
```bash
./scripts/test.sh
```

## Integration with L37

ValidatorAgent receives output from L37's RetrieverAgent:
```python
# L37 output
retrieved_docs = [
  {"id": "doc1", "content": "...", "score": 0.95},
  {"id": "doc2", "content": "...", "score": 0.92}
]

# L38 input
validated_docs = await validator_service.validate_documents(retrieved_docs)

# L38 output (enhanced)
{
  "overall_consistency": 0.85,
  "documents": [
    {
      "id": "doc1",
      "validation_status": "passed",
      "consistency_score": 0.89,
      "contradiction_flags": []
    }
  ]
}
```

## Preparation for L39

L39 will extend this validation infrastructure to check compliance against rules:

- Same architecture (service, cache, dashboard)
- Different prompt: "Does doc violate Rule X?" vs "Do docs contradict?"
- Rule-based scoring vs pairwise comparison

## Technology Stack

- **Backend**: Python 3.12, FastAPI, Gemini AI
- **Frontend**: React 18, Axios, CSS3
- **Cache**: Redis
- **Containerization**: Docker, Docker Compose

## License

Part of VAIA 90-Lesson Curriculum - Module 5: Advanced RAG Orchestration
