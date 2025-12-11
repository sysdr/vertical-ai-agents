#!/bin/bash

API_URL="http://localhost:8000"

case "$1" in
  health)
    echo "=== Health Check ==="
    curl -s "$API_URL/health" | python3 -m json.tool
    ;;
  
  process)
    echo "=== Test Async Processing ==="
    curl -s -X POST "$API_URL/process" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_id": "agent-test1234",
        "prompts": ["What is async/await?", "Explain decorators"],
        "temperature": 0.7,
        "max_tokens": 1024
      }' | python3 -m json.tool
    ;;
  
  metrics)
    echo "=== System Metrics ==="
    curl -s "$API_URL/metrics" | python3 -m json.tool
    ;;
  
  validation)
    echo "=== Test Pydantic Validation ==="
    echo "Sending invalid agent_id..."
    curl -s -X POST "$API_URL/process" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_id": "invalid-id",
        "prompts": ["test"],
        "temperature": 0.7,
        "max_tokens": 1024
      }' | python3 -m json.tool
    ;;
  
  load_test)
    echo "=== Load Test (10 concurrent requests) ==="
    for i in {1..10}; do
      curl -s -X POST "$API_URL/process" \
        -H "Content-Type: application/json" \
        -d "{
          \"agent_id\": \"agent-load${i}0\",
          \"prompts\": [\"Test $i\"],
          \"temperature\": 0.7,
          \"max_tokens\": 100
        }" > /dev/null &
    done
    wait
    echo "Load test complete. Check metrics at $API_URL/metrics"
    ;;
  
  *)
    echo "Usage: $0 {health|process|metrics|validation|load_test}"
    exit 1
    ;;
esac
