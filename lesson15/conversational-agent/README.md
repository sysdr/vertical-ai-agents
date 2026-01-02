# L15: Basic Conversational Agent

Production-ready CLI conversational agent with persistent memory and goal-seeking behavior.

## Features
- Persistent conversation state across sessions
- Multi-user support with isolated conversations
- Goal tracking and completion detection
- Real-time monitoring dashboard
- SQLite persistence layer
- Gemini AI integration

## Quick Start

```bash
# Build
./scripts/build.sh

# Start (API + Dashboard)
./scripts/start.sh

# Or use CLI directly
cd backend
source venv/bin/activate
python cli.py

# Test
./scripts/test.sh
```

## Usage

### CLI Commands
- `/goal <description>` - Set a conversation goal
- `/quit` - Exit

### API Endpoints
- `POST /conversations` - Create conversation
- `POST /messages` - Send message
- `GET /conversations/{id}/history` - Get history

### Dashboard
Visit http://localhost:3000 for real-time monitoring

## Architecture
- **ConversationEngine**: Orchestrates message flow
- **MemoryManager**: SQLite persistence
- **GoalTracker**: LLM-based goal evaluation
- **GeminiClient**: API integration with retry logic

## Testing
```bash
cd backend
source venv/bin/activate
pytest tests/
```
