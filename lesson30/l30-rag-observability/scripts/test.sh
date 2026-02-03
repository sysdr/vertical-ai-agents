#!/bin/bash
echo "Running L30 tests..."

# Ensure we're in project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Wait for backend to be ready (max 60s)
echo "Waiting for backend..."
for i in {1..30}; do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Backend not responding on port 8000. Start services first: ./scripts/start.sh"
    exit 1
  fi
  sleep 2
done

# Test health endpoint
echo "Testing health endpoint..."
curl -sf http://localhost:8000/health || exit 1
echo ""

# Test query endpoint (populates metrics for dashboard)
echo "Testing query endpoint..."
curl -sf -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is observability?"}' | head -c 200
echo ""
echo ""

# Test stats endpoint
echo "Testing stats endpoint..."
curl -sf http://localhost:8000/api/stats || exit 1
echo ""

echo "✅ All tests passed!"
