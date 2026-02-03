# L30: Introduction to Observability for RAG

Production-grade observability layer for RAG systems with comprehensive instrumentation, real-time metrics, and trace visualization.

## Features

- 📊 Real-time performance metrics dashboard
- 🔍 Distributed tracing with trace ID propagation
- 📝 Structured JSON logging
- ⚡ WebSocket-based live metrics streaming
- 💰 Token usage and cost tracking
- 🎯 Latency percentile monitoring (P50, P95, P99)
- 📈 Time-series metrics storage

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
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload &

cd ../frontend
npm install
npm start
```

## Access Points

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Metrics WebSocket**: ws://localhost:8000/ws/metrics

## Architecture

The system implements the observability triangle:

1. **Logs**: Structured JSON logs with trace context
2. **Metrics**: Aggregated performance measurements
3. **Traces**: End-to-end request flow tracking

## Key Metrics

- `embedding_latency_ms`: Embedding generation time
- `retrieval_latency_ms`: Vector search time
- `generation_latency_ms`: LLM response time
- `total_latency_ms`: End-to-end request latency
- `llm_cost_usd`: Per-request token cost
- `chunks_retrieved`: Retrieved context chunks
- `avg_similarity_score`: Retrieval quality metric

## Testing

```bash
./scripts/start.sh   # Start services first (Docker preferred)
./scripts/test.sh    # Run API tests
./scripts/demo.sh    # Run demo queries to populate dashboard metrics
```

**Note**: The dashboard shows "—" for metrics until you run queries. Use the Query Interface or `./scripts/demo.sh` to populate metrics.

## Architecture Integration

Builds on L29's evaluation framework by adding continuous monitoring. Prepares for L31's ReAct planning by establishing trace infrastructure for multi-step reasoning.

## Production Considerations

- Async logging with minimal overhead (<8ms)
- Batched metrics export (10s intervals)
- In-memory aggregation for fast queries
- SQLite for simple deployments (use TimescaleDB/InfluxDB in production)
- WebSocket streaming for real-time dashboards
- Trace sampling for high-volume scenarios

