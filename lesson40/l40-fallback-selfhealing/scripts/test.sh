#!/bin/bash

# Use jq if available for JSON formatting
JQ_CMD="jq . 2>/dev/null || python3 -m json.tool 2>/dev/null || cat"

echo "Testing L40 Fallback & Self-Healing..."

# Wait for backend to be ready
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "Backend ready."
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "ERROR: Backend not ready after 30s"
        exit 1
    fi
done

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | eval $JQ_CMD

# Test query endpoint
echo -e "\nTesting query endpoint..."
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is fallback logic?", "simulate_failure": false}' | eval $JQ_CMD

# Test with simulated failure
echo -e "\nTesting with simulated failure..."
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Test retry mechanism", "simulate_failure": true}' | eval $JQ_CMD

# Test metrics
echo -e "\nTesting metrics endpoint..."
curl -s http://localhost:8000/metrics | eval $JQ_CMD

echo -e "\n✅ Tests complete!"
