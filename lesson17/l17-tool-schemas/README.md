# L17: Designing Robust Tool Schemas

Production-grade tool schema validation system with Pydantic and Gemini AI.

## Quick Start

```bash
# Setup and build
cd l17-tool-schemas
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Features

- **Pydantic Schema Validation**: Type-safe tool parameter validation
- **Gemini AI Integration**: Function calling with validated schemas
- **Interactive Dashboard**: Real-time validation testing
- **Metrics Dashboard**: Track tool usage and performance
- **Production Patterns**: Error handling, schema registry, type safety

## Architecture

- **Backend**: FastAPI + Pydantic v2 + Gemini AI
- **Frontend**: React with validation UI
- **Validation**: Comprehensive input/output validation
- **Error Recovery**: Structured error responses

## Endpoints

- `GET /tools` - List registered tools
- `POST /validate` - Validate tool parameters
- `POST /query` - Process natural language queries
- `GET /health` - Health check
- `GET /metrics` - Dashboard metrics

## Usage

1. Open http://localhost:3000
2. Try query: "What's the weather in Paris?"
3. View validation in real-time
4. Test schemas with custom parameters
5. Check metrics dashboard for usage statistics

## Testing

Tests cover:
- Valid parameter validation
- Invalid parameter rejection
- Field validators
- Range constraints
- Type coercion

## Next Steps

L18 builds ToolExecutor class for dynamic tool execution based on these validated schemas.
