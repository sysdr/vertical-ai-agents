#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting L36 PlannerAgent services..."

# Start backend
echo "Starting FastAPI backend..."
cd "$SCRIPT_DIR/backend"
if [ -d "venv" ] && venv/bin/python -c "import uvicorn" 2>/dev/null; then
  venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
else
  PYTHONPATH=".packages:." python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend
echo "Waiting for backend to be ready..."
sleep 3

# Start frontend
echo "Starting React frontend..."
cd "$SCRIPT_DIR/frontend"
PORT=3000 npm start &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "=== L36 PlannerAgent Running ==="
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
