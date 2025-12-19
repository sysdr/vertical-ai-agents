# L8: Core Agent Theory - Decision Maker

## Overview
Complete implementation of autonomous AI agent with:
- Decision-making engine with deliberation
- Planning component for goal decomposition
- Tool registry system
- State machine tracking
- Real-time monitoring dashboard

## Quick Start

### Non-Docker
```bash
./build.sh      # Install dependencies
./start.sh      # Start services
./test.sh       # Run tests
./stop.sh       # Stop services
```

### Docker
```bash
docker-compose up
```

## Architecture

### Components
1. **Decision Maker**: Core deliberation engine
2. **Planner**: Decomposes goals into steps
3. **Tool Registry**: Manages available tools
4. **State Tracker**: Monitors agent state transitions

### Integration with L7
- Reuses `gemini_json_call()` for structured LLM responses
- Extends JSON prompting with autonomous decision-making
- Adds planning and tool orchestration layers

## API Endpoints

- `POST /agent/execute` - Execute agent goal
- `GET /tools` - List available tools
- `GET /metrics` - System metrics
- `GET /health` - Health check
- `WS /ws` - WebSocket updates

## Testing

Execute test goal:
```bash
curl -X POST http://localhost:8000/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Calculate 5 factorial",
    "context": {},
    "max_steps": 10,
    "timeout_seconds": 30
  }'
```

Expected response includes:
- Generated plan
- Execution traces
- Tool results
- Performance metrics

## Dashboard

Access at http://localhost:3000

Features:
- Execute agent goals
- View decision traces
- Monitor available tools
- Track system metrics

## Production Considerations

- Decision latency: <500ms per step
- Tool selection accuracy: 95%+
- Complete trace logging
- Error recovery with exponential backoff
- WebSocket for real-time updates

## Next Steps (L9)

L9 adds persistent memory:
- Redis for short-term context
- PostgreSQL for long-term storage
- User preference learning
- Personalized decision-making

## Troubleshooting

**Backend won't start:**
```bash
cd backend
source venv/bin/activate
python main.py
# Check error output
```

**Frontend issues:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

**Port conflicts:**
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```
