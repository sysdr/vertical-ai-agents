# L6: Interacting with LLM APIs - Production-Grade Integration

## Quick Start

### Option 1: Local Development
```bash
# Build
./build.sh

# Start
./start.sh

# Access
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs

# Stop
./stop.sh

# Test
./test.sh
```

### Option 2: Docker
```bash
docker-compose up --build
```

## Features

- ✅ Token bucket rate limiting (RPM + TPM)
- ✅ Circuit breaker pattern for fault tolerance
- ✅ Exponential backoff with jitter
- ✅ Real-time cost tracking
- ✅ Structured logging
- ✅ Production-grade async I/O
- ✅ Comprehensive testing

## Architecture

```
React UI (3000) ──→ FastAPI (8000) ──→ Gemini AI
                         │
                         ├─ Rate Limiter
                         ├─ Circuit Breaker
                         └─ Cost Tracker
```

## Testing

```bash
# All tests
./test.sh

# Specific tests
./test.sh rate_limit
./test.sh circuit_breaker
./test.sh cost_tracking
```

## Configuration

Edit `.env` to customize:
- Rate limits (RPM/TPM)
- Circuit breaker thresholds
- Cost parameters
- Model selection

## Next Steps

This foundation prepares you for:
- L7: Basic LLM Prompting & JSON Output
- L8: Streaming Responses
- L15: Multi-Model Orchestration
