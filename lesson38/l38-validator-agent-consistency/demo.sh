#!/bin/bash
# L38 ValidatorAgent demo: ensure services run, call validate API, print dashboard URL
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
API_URL="${API_URL:-http://localhost:8000/api/v1}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:3000}"

echo "=========================================="
echo "L38 ValidatorAgent - Demo"
echo "=========================================="

# Start services if backend is not responding
if ! curl -sf "$API_URL/../" >/dev/null 2>&1; then
  echo "Backend not detected. Starting services..."
  "$SCRIPT_DIR/start.sh"
  echo "Waiting for backend..."
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf "$API_URL/../" >/dev/null 2>&1; then break; fi
    sleep 2
  done
fi

# Health check
if ! curl -sf "$API_URL/validator/health" >/dev/null 2>&1; then
  echo "Backend not ready. Run ./start.sh and ensure backend venv exists (./build.sh)."
  echo "Dashboard: $DASHBOARD_URL"
  exit 1
fi

echo "Backend is up. Running validation demo..."
RESPONSE=$(curl -sf -X POST "$API_URL/validator/validate" \
  -H "Content-Type: application/json" \
  -d '[
    {"id":"doc1","content":"Our Q3 revenue was $50 million, up 20% from Q2.","score":0.95,"source":"Financial Report","metadata":{}},
    {"id":"doc2","content":"Third quarter earnings reached $50M, showing strong growth.","score":0.92,"source":"Press Release","metadata":{}},
    {"id":"doc3","content":"Q3 revenue totaled $48 million according to preliminary figures.","score":0.88,"source":"Internal Memo","metadata":{}}
  ]' 2>/dev/null) || true

if [ -n "$RESPONSE" ]; then
  echo ""
  echo "Validation result (summary):"
  echo "$RESPONSE" | python3 -c "
import sys, json
try:
  d = json.load(sys.stdin)
  print('  Overall consistency: {:.1%}'.format(d.get('overall_consistency', 0)))
  print('  Pairs validated: {}/{}'.format(d.get('validated_pairs', 0), d.get('total_pairs', 0)))
  print('  Cache hit rate: {:.1%}'.format(d.get('cache_hit_rate', 0)))
  print('  Processing time: {} ms'.format(d.get('processing_time_ms', 0)))
  for doc in d.get('documents', [])[:5]:
    print('  - {}: {} (consistency: {:.1%})'.format(doc.get('id'), doc.get('validation_status'), doc.get('consistency_score', 0)))
except Exception as e:
  print('  (raw):', sys.stdin.read()[:500])
" 2>/dev/null || echo "  $RESPONSE"
else
  echo "Validation request failed (backend may still be starting). Try: curl -X POST $API_URL/validator/validate -H 'Content-Type: application/json' -d '[{\"id\":\"a\",\"content\":\"x\",\"score\":0.9,\"source\":\"s\",\"metadata\":{}},{\"id\":\"b\",\"content\":\"y\",\"score\":0.9,\"source\":\"s\",\"metadata\":{}}]'"
fi

echo ""
echo "Dashboard: $DASHBOARD_URL"
echo "API docs:  http://localhost:8000/docs"
echo "=========================================="
