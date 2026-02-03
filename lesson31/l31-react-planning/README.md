# L31: Deep Dive into ReAct Planning

Enterprise-grade ReAct planning system implementing the Observe→Thought→Action→Observe loop with LLM-powered adaptive decision-making.

## Features

- **ReAct Planning Engine**: Iterative reasoning loop with structured thought generation
- **LLM Integration**: Gemini AI powered reasoning and action generation
- **Observability**: Comprehensive metrics tracking token usage, latency, and decision quality
- **Interactive Dashboard**: Real-time visualization of planning traces and reasoning steps
- **Production Patterns**: Confidence scoring, termination conditions, and error handling

## Quick Start

```bash
# Build
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop
./stop.sh
```

## Access Points

- Frontend Dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture

The system consists of:
- **ReActPlanner**: Orchestrates the planning loop
- **ReasoningEngine**: Interfaces with Gemini AI
- **ObservationStore**: Maintains planning context
- **MetricsCollector**: Tracks performance metrics

## Usage

1. Open dashboard at http://localhost:3000
2. Enter a task (e.g., "Analyze database performance issues")
3. Set max iterations (default: 10)
4. Click "Execute Planning"
5. View the reasoning trace with thoughts, actions, and confidence scores

## Integration

- Builds on L30's observability infrastructure
- Prepares foundation for L32's tool-equipped agents
- Part of Module 3: Agentic Reasoning and Planning

## Requirements

- Python 3.12+
- Node.js 20+
- Docker & Docker Compose (optional)
- Gemini AI API access
