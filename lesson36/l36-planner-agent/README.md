# L36: The Planner Agent - Query Decomposition

Production PlannerAgent that decomposes complex queries into optimized sub-queries for Agentic RAG systems.

## Quick Start

```bash
# Build
./build.sh

# Start services
./start.sh

# Open browser
open http://localhost:3000

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Architecture

- **Backend**: FastAPI + Gemini AI (port 8000)
- **Frontend**: React dashboard (port 3000)
- **AI Model**: Gemini 2.0 Flash for query decomposition

## Features

- ✓ LLM-powered query decomposition
- ✓ Structured JSON validation with Pydantic
- ✓ Parallel vs sequential strategy planning
- ✓ Fallback to original query on failures
- ✓ Real-time UI showing decomposition results
- ✓ Production-ready with timeout protection

## Integration

Builds on: L35 (Agentic RAG architecture)
Prepares for: L37 (RetrieverAgent with reranking)

## Validation

Test complex query: "Compare Tesla's autopilot to Mercedes' system and analyze regulatory concerns"

Expected: 3-4 sub-queries with parallel strategy
