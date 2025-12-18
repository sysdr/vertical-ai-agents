# L7: Basic LLM Prompting & JSON Output

Production-grade prompt engineering system with multi-strategy JSON parsing and validation.

## Features

- ✅ Smart prompt construction with schema enforcement
- ✅ Multi-strategy JSON parsing (direct, regex, LLM repair)
- ✅ Real-time metrics dashboard
- ✅ Schema validation with Pydantic
- ✅ Error recovery and observability

## Quick Start

### Option 1: Local Development

```bash
# Build
./build.sh

# Start system
./start.sh

# Run tests
./test.sh

# Stop
./stop.sh
```

### Option 2: Docker

```bash
# Build containers
./docker-build.sh

# Start
./docker-start.sh
```

## Access

- Frontend Dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Testing the System

1. Open the dashboard at http://localhost:3000
2. Try the "Test: User Profile" button for quick demo
3. Customize instructions and schemas
4. Watch real-time metrics update

## API Endpoints

- `POST /generate` - Generate structured JSON from prompt
- `GET /metrics` - Get parsing metrics
- `POST /test/user-profile` - Test user profile generation
- `POST /test/product-info` - Test product info generation

## Architecture

- **Backend**: FastAPI with Gemini AI integration
- **Frontend**: React with real-time updates
- **Parsing**: 3-tier fallback strategy
- **Validation**: Pydantic schema enforcement

## Key Concepts

### Parse Strategies

1. **Direct Parse** - Standard json.loads()
2. **Regex Extraction** - Extract from markdown/text
3. **LLM Repair** - Self-correction by the LLM

### Success Metrics

- Direct parse success rate > 90%
- Overall success rate > 95%
- Average parse time < 100ms

## Troubleshooting

**Backend won't start**: Check that port 8000 is free
**Frontend won't start**: Check that port 3000 is free
**CORS errors**: Ensure backend is running first
**Parse failures**: Check prompt structure and schema format

## Next Steps

This lesson prepares for L8 (Core Agent Theory) by providing:
- Structured output parsing for agent perception
- Validation pipelines for decision-making
- Error recovery patterns for production agents
