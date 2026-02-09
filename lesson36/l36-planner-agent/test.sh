#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Testing L36 PlannerAgent..."

cd backend
if [ -d "venv" ] && venv/bin/python -c "import pytest" 2>/dev/null; then
  PYTHON_CMD="venv/bin/python"
elif [ -d ".packages" ]; then
  export PYTHONPATH=".packages:.:$PYTHONPATH"
  PYTHON_CMD="python3"
else
  PYTHON_CMD="python3"
fi

# Run pytest
$PYTHON_CMD -m pytest tests/ -v

# Test API endpoint
echo ""
echo "Testing API endpoint..."
curl -s -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare machine learning to deep learning", "max_sub_queries": 3}' \
  2>/dev/null | $PYTHON_CMD -m json.tool 2>/dev/null || echo "API test requires running backend"

cd ..

echo ""
echo "Tests complete!"
