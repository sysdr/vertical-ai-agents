# L35: Architecting Agentic RAG

Multi-agent RAG orchestrator with Planner, Retriever, Validator, and Synthesizer agents.

## Quick Start

### Non-Docker
```bash
./build.sh
./start.sh
```

Visit http://localhost:3000

### Docker
```bash
docker-compose up --build
```

## Architecture

- **PlannerAgent**: Analyzes queries, creates retrieval plans
- **RetrieverAgent**: Executes document searches
- **ValidatorAgent**: Scores factual consistency
- **SynthesizerAgent**: Generates responses with citations

## Testing
```bash
./test.sh
```

## Stop
```bash
./stop.sh  # or Ctrl+C
```
