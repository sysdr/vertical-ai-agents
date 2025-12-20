# L9: Agent Memory Implementation

## Overview
Dual-tier memory system for AI agents:
- **Short-term**: In-memory dict for session context (fast)
- **Long-term**: File-based JSON for persistent storage

## Quick Start

### Docker (Recommended)
```bash
docker-compose up --build
```

### Local Development
```bash
./build.sh    # Install dependencies
./start.sh    # Start backend + frontend
./test.sh     # Run verification tests
```

## Access Points
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Architecture

### Short-Term Memory
- Session-scoped message history
- Last 20 messages retained
- In-memory dict (< 10ms retrieval)

### Long-Term Memory
- User-scoped persistent facts
- JSON file storage
- Keyword search capability

## API Endpoints

### Store Message
```bash
POST /api/store
{
  "session_id": "session_001",
  "role": "user",
  "content": "I like pizza"
}
```

### Recall Session
```bash
GET /api/recall/{session_id}?limit=10
```

### Store Long-Term Fact
```bash
POST /api/longterm/store
{
  "user_id": "user_001",
  "fact": {"preference": "pizza"}
}
```

### Search Long-Term
```bash
POST /api/longterm/search
{
  "user_id": "user_001",
  "keywords": ["pizza"],
  "limit": 5
}
```

## Production Notes
- Replace file storage with Redis (short-term) + PostgreSQL (long-term)
- Add vector embeddings for semantic search
- Implement async writes for high throughput
- Add memory eviction policies
- Monitor memory usage per session

## Next Steps
L10 will combine this memory system with L8's decision_maker to build SimpleAgent class.
