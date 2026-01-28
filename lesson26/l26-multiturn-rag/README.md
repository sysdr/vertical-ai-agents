# L26: Handling Ambiguity & Multi-Turn RAG

Production-grade conversational RAG system with context management, ambiguity detection, and multi-turn Q&A.

## Features

- **Conversation Memory**: Configurable window strategies (sliding, token-based)
- **Ambiguity Detection**: Linguistic patterns + retrieval confidence scoring
- **Context-Aware Retrieval**: Query expansion using conversation history
- **Clarification Engine**: Generates targeted questions for ambiguous queries
- **Redis Persistence**: Session management with 24-hour TTL
- **Real-time Dashboard**: Track conversation state, tokens, and ambiguity scores

## Quick Start

### Non-Docker Setup

```bash
# Build
./scripts/build.sh

# Start (opens http://localhost:3000)
./scripts/start.sh

# Test
./scripts/test.sh

# Stop
./scripts/stop.sh
```

### Docker Setup

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Architecture

- **Backend**: FastAPI + Gemini AI + ChromaDB + Redis
- **Frontend**: React with real-time conversation UI
- **Memory**: ConversationBufferMemory with multiple window strategies
- **Detection**: Multi-factor ambiguity scoring
- **Expansion**: LLM-powered query expansion with context

## Testing Multi-Turn Conversations

```python
# Test ambiguous queries
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "query": "Tell me more about that"
  }'

# Test context-aware retrieval
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "query": "What are the pricing plans?"
  }'

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "query": "How much does the enterprise one cost?"
  }'
```

## Configuration

Edit `backend/.env`:
- `MAX_CONVERSATION_TURNS`: Memory window size (default: 10)
- `MAX_CONVERSATION_TOKENS`: Token limit (default: 2000)
- `AMBIGUITY_THRESHOLD`: Detection threshold (default: 0.4)
- `SESSION_TTL_HOURS`: Redis TTL (default: 24)

## From Previous Lesson (L25)

Builds on L25's system prompting patterns:
- Grounding instructions now applied across conversation turns
- Prompt templates extended with conversation history
- Relevance scoring integrated into ambiguity detection

## Preparing for Next Lesson (L27)

Sets foundation for RAG evaluation:
- Logs ambiguity scores for groundedness metrics
- Tracks retrieval confidence for relevance scoring
- Instruments conversation quality signals
- Provides test data for Ragas evaluation library
