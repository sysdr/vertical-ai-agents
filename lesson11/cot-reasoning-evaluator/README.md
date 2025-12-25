# L11: Chain-of-Thought Reasoning Evaluator

Production-grade CoT prompting system with automated reasoning quality analysis.

## Quick Start

```bash
# Build
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Access

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- Chain-of-Thought prompt construction
- Automated reasoning trace parsing
- Multi-dimensional quality scoring
- Visual reasoning analysis dashboard
- Persistent memory for trace storage
- Real-time quality statistics

## Architecture

- **Backend**: Python FastAPI + Gemini AI
- **Frontend**: React with responsive UI
- **Persistence**: JSON-based memory storage
- **Deployment**: Docker + non-Docker modes

## Testing Sample Queries

1. Math: "If Alice has 5 cookies and gives 2 to Bob, then Bob gives 1 to Charlie, how many cookies does each person have?"
2. Logic: "If all cats are mammals and all mammals are animals, are all cats animals?"
3. Planning: "How would you organize a 3-day conference for 200 attendees?"

## Integration with L10

Extends SimpleAgent from L10 with `reason_with_cot()` method and quality evaluation capabilities.

## Prepares for L12

High-quality reasoning traces stored in memory become few-shot examples for L12's prompting techniques.
