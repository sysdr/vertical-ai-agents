# VAIA L5: Model Landscape & Selection

## Overview
Production-grade model comparison and benchmarking platform for evaluating LLM performance across multiple dimensions.

## Quick Start
```bash
# Build
./scripts/build.sh

# Start services
./scripts/start.sh

# Test (in new terminal)
./scripts/test.sh

# Stop
./scripts/stop.sh
```

## Architecture
- **Backend**: FastAPI with async model clients
- **Frontend**: React with Recharts visualization
- **Models**: Gemini Flash, Pro, Ultra comparison

## Features
- Real-time benchmarking across models
- Latency and cost profiling
- Pareto frontier analysis
- Smart model recommendations
- Interactive dashboard

## API Endpoints
- `GET /api/models` - Available models
- `POST /api/benchmark` - Run benchmark
- `POST /api/compare` - Compare models
- `GET /api/recommendations` - Get suggestions

## Learning Objectives
- Model selection criteria
- Performance vs cost trade-offs
- Benchmark methodology
- Production deployment strategies

## Integration
- Builds on L4 parameter analysis
- Prepares for L6 secure API access
- Part of 90-lesson VAIA curriculum
