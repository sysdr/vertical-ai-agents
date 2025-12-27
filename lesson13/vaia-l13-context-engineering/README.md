# VAIA L13: Context Engineering

Production-grade context management and optimization for enterprise AI agents.

## Overview

This lesson implements intelligent context window management using:
- Token counting with tiktoken
- Multi-strategy summarization (extractive, abstractive, hybrid)
- Automatic context optimization
- Real-time cost analysis

## Quick Start

### Using Docker (Recommended)
```bash
./build.sh
./start.sh
```

### Local Development
```bash
# Install dependencies
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Architecture

- **Backend**: FastAPI + Gemini AI + tiktoken
- **Frontend**: React with real-time monitoring
- **Context Manager**: Intelligent optimization pipeline

## Features

1. **Token Counter**: Precise token analysis with recommendations
2. **Summarizer**: Multiple compression strategies
3. **Context Optimizer**: Automatic strategy selection and optimization
4. **Dashboard**: Real-time metrics and cost savings

## API Endpoints

- `POST /api/v1/count-tokens` - Analyze token usage
- `POST /api/v1/summarize` - Compress text
- `POST /api/v1/optimize-context` - Intelligent optimization
- `GET /health` - Service health check

## Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Building on L12

Extends few-shot prompting with context management for token-efficient examples.

## Enabling L14

Provides context optimization layer for state serialization and persistence.

## License

Part of VAIA 90-Lesson Curriculum
