#!/bin/bash
# Run evaluation via API to populate dashboard metrics
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running demo evaluation..."
curl -s -X POST http://localhost:8000/api/evaluate/run \
  -H "Content-Type: application/json" \
  -d '{"test_suite_name":"default","iterations":3}' | python3 -m json.tool

echo ""
echo "Metrics summary:"
curl -s http://localhost:8000/api/metrics/summary | python3 -m json.tool
