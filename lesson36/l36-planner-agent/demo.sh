#!/bin/bash
# Demo script: Run query decompositions to populate metrics dashboard
# Run after starting services: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
API_URL="http://localhost:8000"

echo "L36 PlannerAgent Demo"
echo "===================="
echo ""

# Wait for backend
echo "Waiting for backend..."
for i in $(seq 1 30); do
  if curl -sf "$API_URL/health" >/dev/null 2>&1; then
    echo "Backend ready."
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Backend not responding. Start with: ./start.sh"
    exit 1
  fi
  sleep 2
done

echo ""
echo "Running demo queries to populate dashboard metrics..."
echo ""

queries=(
  "Compare machine learning to deep learning"
  "What are the differences between RAG and fine-tuning for LLMs?"
  "Explain quantum computing applications in cryptography and drug discovery"
)

for q in "${queries[@]}"; do
  echo "Query: $q"
  curl -sf -X POST "$API_URL/plan" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$q\", \"max_sub_queries\": 4}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('success') and d.get('plan'):
  p = d['plan']
  n = len(p.get('sub_queries', []))
  print(f'  Sub-queries: {n}, Strategy: {p.get(\"strategy\", \"?\")}, Latency: {d.get(\"processing_time_ms\", 0):.0f}ms')
else:
  print(f'  Error: {d.get(\"error\", \"unknown\")}')
" 2>/dev/null || echo "  (response received)"
  echo ""
  sleep 1
done

echo "Demo complete! Check the dashboard at http://localhost:3000"
echo "Metrics should now show non-zero values (Total Queries, Avg Latency, etc.)."
