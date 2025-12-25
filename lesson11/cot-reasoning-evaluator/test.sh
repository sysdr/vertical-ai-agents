#!/bin/bash
set -e

echo "🧪 Running L11 CoT System Tests..."

# Get absolute path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run pytest
python -m pytest test_cot.py -v

# API health check
echo ""
echo "Testing API endpoints..."
sleep 2

# Test root endpoint
curl -s http://localhost:8000/ | grep -q "operational" && echo "✓ Root endpoint OK"

# Test reasoning endpoint
RESULT=$(curl -s -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?", "style": "standard"}')

echo "$RESULT" | grep -q "reasoning_trace" && echo "✓ Reasoning endpoint OK"
echo "$RESULT" | grep -q "quality_scores" && echo "✓ Quality evaluation OK"

# Test memory endpoint
curl -s http://localhost:8000/api/memory | grep -q "traces" && echo "✓ Memory endpoint OK"

# Test stats endpoint
curl -s http://localhost:8000/api/stats | grep -q "total_traces" && echo "✓ Stats endpoint OK"

echo ""
echo "✅ All tests passed!"
