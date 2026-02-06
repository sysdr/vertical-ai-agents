# L34: Planning Loop Controls & Budgeting

Budget-aware ReAct agent with mandatory iteration limits, token budgets, and cost monitoring.

## Quick Start

### With Docker (Recommended)
```bash
./scripts/start.sh
./scripts/test.sh
```

### Without Docker
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_api_key_here
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm start
```

## Access Points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Features

- ✅ Hard iteration limits preventing runaway execution
- ✅ Token-per-turn budgets with real-time tracking
- ✅ Environment-specific policies (dev/staging/prod)
- ✅ Exception-based circuit breakers
- ✅ Real-time monitoring dashboard
- ✅ Graceful degradation on budget violations

## Architecture

The system implements three-tier budget enforcement:
1. **BudgetManager**: Tracks usage and enforces limits
2. **ExecutionController**: Coordinates ReAct/Reflexion loops with budget checks
3. **Monitoring Dashboard**: Real-time metrics visualization

## Environment Policies

- **Development**: 50 iterations, 500K tokens, warn mode
- **Staging**: 20 iterations, 100K tokens, warn mode
- **Production**: 10 iterations, 50K tokens, strict mode

## Testing

Run the test suite:
```bash
./scripts/test.sh
```

Test with custom budgets:
```bash
curl -X POST http://localhost:8000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Research and summarize quantum computing",
    "environment": "production",
    "max_iterations": 5,
    "max_tokens": 25000
  }'
```

## Stop Services

```bash
./scripts/stop.sh
```

## Next Steps

L35: Agentic RAG will use these budget controls for multi-agent systems where each component (Planner, Retriever, Validator, Synthesizer) has independent budgets.
