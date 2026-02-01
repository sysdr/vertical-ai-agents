# L28: Tool-Equipped Agent Build

Enterprise-grade agent combining RAG, tools, and evaluation.

## API Key Setup

**Required:** Set your Gemini API key in `backend/.env`:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set GEMINI_API_KEY=your_key
```

Then restart: `./stop.sh && ./start.sh`

Get an API key at: https://aistudio.google.com/apikey

## Quick Start

### Option 1: Automated Setup
```bash
./build.sh
./start.sh
```

### Option 2: Docker
```bash
docker-compose up
```

### Option 3: Manual Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm start
```

### Option 4: CLI Only
```bash
cd backend
source venv/bin/activate
python cli.py
```

## Access Points

- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **CLI**: `python backend/cli.py`

## Features

- Intent classification (RAG / Tool / Hybrid)
- Weather, calculator, database tools
- Real-time evaluation metrics
- Conversation history
- Production-grade patterns

## Testing

```bash
./test.sh
```

## Example Queries

- "What is Python?" (RAG)
- "What's the weather in Tokyo?" (Tool)
- "Calculate 2^10" (Tool)
- "Query users from the database" (Tool)
- "Search for Python info and calculate its popularity score" (Hybrid)
