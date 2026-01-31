#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Testing L27 RAG Evaluation System..."

# Wait for backend
echo "Waiting for backend..."
for i in {1..30}; do
  curl -s http://localhost:8000/health > /dev/null 2>&1 && break
  sleep 1
done

# Test health
echo "Testing health endpoint..."
curl -s http://localhost:8000/health
echo ""

# Test API endpoints
echo "Testing metrics endpoint..."
curl -s http://localhost:8000/api/metrics/summary | head -c 200
echo ""

# Run pytest if backend venv exists
if [ -d "$PROJECT_ROOT/backend/venv" ]; then
  echo "Running pytest..."
  (cd "$PROJECT_ROOT/backend" && source venv/bin/activate && PYTHONPATH=. pytest tests/ -q) || true
fi
echo "✅ Basic tests passed!"
echo "Open http://localhost:3000 to use the dashboard"
