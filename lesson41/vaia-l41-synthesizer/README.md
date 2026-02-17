# VAIA L41: The Synthesizer Agent & XAI

Enterprise-grade response synthesis with Chain-of-Thought reasoning and explicit citations.

## Quick Start

1. **Set your Gemini API key** in `.env`: `GEMINI_API_KEY=your_key` (get one at https://aistudio.google.com/apikey). Dashboard demo and synthesis need it; without it, health checks still pass and the dashboard will show zeros until a successful run.

### Non-Docker Setup
```bash
./build.sh    # Install dependencies
./start.sh    # Start services (backend from project root with PYTHONPATH)
./test.sh     # Run tests (health + demo; demo requires valid API key)
./stop.sh     # Stop services
```

### Docker Setup
```bash
docker-compose up -d
docker-compose logs -f
```

## Access Points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- ✅ Chain-of-Thought reasoning engine
- ✅ Automatic citation mapping
- ✅ Coherence validation
- ✅ Adaptive XAI formatting
- ✅ Structured logging for L42
- ✅ Production-ready architecture

## Architecture

```
Query → CoT Engine → Citation Mapper → Validator → XAI Formatter → Response
         (3-7 steps)   (Source links)    (Quality)   (Transparency)
```

## API Usage

```python
import requests

response = requests.post("http://localhost:8000/synthesize", json={
    "query": "How does the Synthesizer work?",
    "validated_chunks": [
        {
            "id": "chunk_1",
            "content": "The Synthesizer uses CoT reasoning...",
            "source": "architecture.pdf",
            "quality_score": 0.9
        }
    ]
})

print(response.json()["response_text"])
```

## Lesson 42 Preview

The structured logs and reasoning chains created here feed directly into L42's traceability layer for comprehensive audit trails.

## Assignment

Implement confidence scoring mechanism (see article.md for details).
