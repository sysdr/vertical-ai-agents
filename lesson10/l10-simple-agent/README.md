# L10: Building a Simple Agent

Production-ready autonomous agent with memory integration and goal-seeking behavior.

## Quick Start

### With Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

### Without Docker
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py &

cd ../frontend
npm install
npm start
```

## Access

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- ✅ Goal-driven autonomous behavior
- ✅ Dual-tier memory (short-term + long-term)
- ✅ Self-evaluation and progress tracking
- ✅ Observable decision logging
- ✅ Real-time state machine visualization
- ✅ Production-ready patterns

## Testing

```bash
./scripts/test.sh
```

## Architecture

- **Backend**: FastAPI + Gemini AI + dual-tier memory
- **Frontend**: React dashboard with real-time updates
- **Agent**: SimpleAgent class with IDLE→THINKING→ACTING→EVALUATING→COMPLETE loop

## Stop Services

```bash
./scripts/stop.sh
```
