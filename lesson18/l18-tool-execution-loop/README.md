# L18: Tool Execution Loop

Production-grade tool execution engine with LLM orchestration and real-time monitoring.

## Setup

### 1. Configure API Key

You need a Google Gemini API key to use this application. Get one from: https://makersuite.google.com/app/apikey

**Option 1: Using .env file (Recommended)**
```bash
cd backend
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

**Option 2: Using environment variable**
```bash
export GEMINI_API_KEY=your_api_key_here
# or
export GOOGLE_API_KEY=your_api_key_here
```

## Quick Start

### Non-Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

### Docker
```bash
docker-compose up --build
```

## Access
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Features
- Dynamic function routing and execution
- Multi-turn conversation loops
- Execution safety (timeouts, error handling)
- Real-time monitoring dashboard
- Comprehensive execution logging
- Tool performance metrics

## Architecture
Built on L17 tool schemas, provides foundation for L19 RAG implementation.

## Testing
```bash
./scripts/test.sh
```

## Components
- ToolExecutor: Main execution loop coordinator
- FunctionRegistry: Dynamic tool registration
- SafeExecutor: Timeout and error wrapper
- Real-time WebSocket monitoring
