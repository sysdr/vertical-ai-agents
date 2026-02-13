# L40: Fallback Logic & Self-Healing

## Overview
Implements Validator ↔ Planner feedback loops with circuit breaker patterns for self-healing VAIA systems.

## Quick Start

### With Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

### Without Docker
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000 &

cd ../frontend
npm install
npm start
```

## Access
- Backend API: http://localhost:8000
- Frontend Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Testing
```bash
./scripts/test.sh
```

## Architecture
- FallbackOrchestrator: Manages retry loops and circuit breaker
- EnhancedPlannerAgent: Query reformulation based on failure signals
- EnhancedValidatorAgent: Structured validation with failure feedback
- Circuit Breaker: Prevents cascade failures
- Adaptive Retry Policy: Exponential backoff with jitter

## Features
- 3-tier retry logic with exponential backoff
- Query reformulation strategies per failure type
- Circuit breaker (closed/open/half-open states)
- Real-time monitoring dashboard
- Graceful degradation on max retries
- Retry history tracking

## Stop Services
```bash
./scripts/stop.sh
```
