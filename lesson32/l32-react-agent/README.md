# L32: Hands-On ReAct Agent Build

Production-grade ReAct agent with multi-tool integration and real-time reasoning visualization.

## Features

- **ReActAgent Class**: Modular agent with tool registry system
- **Multi-Tool Integration**: Wikipedia, Stock Price, Calculator, Weather
- **Real-Time Visualization**: Interactive dashboard showing Thought→Action→Observation cycles
- **Enterprise-Ready**: Error handling, retry logic, comprehensive logging

## Quick Start

### 1. Set up API Key (required)

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
```

See [API_KEY_SETUP.md](API_KEY_SETUP.md) for details.

### 2. With Docker (Recommended)
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
python app.py &

cd ../frontend
npm install
npm start
```

## Access Points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Testing

```bash
./scripts/test.sh
```

## Example Queries

- "What is the current price of Google stock?"
- "Tell me about Python programming language"
- "What is 1234 * 5678?"
- "Compare the stock prices of Google and Apple"

## Architecture

- **Backend**: FastAPI + Python
- **Frontend**: React + JavaScript
- **LLM**: Google Gemini 1.5 Flash
- **Tools**: Wikipedia API, Stock Price (simulated), Calculator, Weather (simulated)

## API Endpoints

- `POST /agent/query` - Execute agent task
- `GET /agent/tools` - List available tools
- `GET /agent/history/{session_id}` - Retrieve reasoning trace
- `GET /health` - Health check

## Components Built

- `ReActAgent` - Main agent class with execution loop
- `ToolRegistry` - Tool management system
- `Tool` base class - Abstract interface for all tools
- 4 Production tools: Wikipedia, StockPrice, Calculator, Weather

## Next Steps (L33)

L33 will add Reflexion (self-correction) to this agent:
- Reflect step after each action
- Self-critique reasoning
- Plan refinement based on reflection
