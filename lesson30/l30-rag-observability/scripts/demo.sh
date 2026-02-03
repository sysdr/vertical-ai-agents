#!/bin/bash
# Demo script: Run queries to populate dashboard metrics
# Run this after starting services to see metrics update on the dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
API_URL="http://localhost:8000"

echo "L30 RAG Observability Demo"
echo "=========================="
echo ""

# Wait for backend
echo "Waiting for backend..."
for i in {1..30}; do
  if curl -sf "$API_URL/health" >/dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Backend not responding. Start with: ./scripts/start.sh"
    exit 1
  fi
  sleep 2
done

echo ""
echo "Running demo queries to populate dashboard metrics..."
echo ""

queries=(
  "What is RAG?"
  "What is observability?"
  "How does vector search work?"
)

for q in "${queries[@]}"; do
  echo "Query: $q"
  curl -sf -X POST "$API_URL/api/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$q\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('metrics',{}); print(f'  Latency: {m.get(\"total_latency_ms\",0):.0f}ms, Chunks: {m.get(\"chunks_retrieved\",0)}')" 2>/dev/null || echo "  (response received)"
  echo ""
  sleep 1
done

echo "Demo complete! Check the dashboard at http://localhost:3000"
echo "Metrics should now show non-zero values."
