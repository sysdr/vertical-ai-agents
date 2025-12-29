# L14: State Management for Agents

Production-grade state persistence for VAIA systems with PostgreSQL and Redis.

## Quick Start

```bash
# Build
./build.sh

# Start (requires Docker)
./start.sh

# Test
./test.sh

# Stop
./stop.sh
```

## Architecture

- **Backend**: FastAPI + PostgreSQL + Redis
- **Frontend**: React dashboard
- **State**: Dual-tier (hot/cold) persistence
- **Versioning**: Automatic state snapshots

## Features

- ✅ Pydantic state models with validation
- ✅ Dual-tier storage (Redis + PostgreSQL)
- ✅ Automatic state versioning
- ✅ State diff and rollback
- ✅ Production error handling
- ✅ Real-time dashboard

## Endpoints

- `POST /api/chat` - Process message with state
- `GET /api/state/{session_id}` - Get current state
- `POST /api/state/diff` - Compare state versions
- `POST /api/state/rollback` - Rollback state

## Access

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Learning Path

Previous: L13 Context Engineering
Current: L14 State Management
Next: L15 Conversational Agent
