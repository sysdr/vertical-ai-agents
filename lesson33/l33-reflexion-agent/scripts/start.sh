#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if exists
[ -f "$PROJECT_ROOT/venv/bin/activate" ] && source "$PROJECT_ROOT/venv/bin/activate"

echo "Starting L33 Reflexion Agent..."
echo "Project root: $PROJECT_ROOT"

# Check for duplicate services
if lsof -i :8000 >/dev/null 2>&1; then
  echo "ERROR: Port 8000 already in use. Run 'bash scripts/stop.sh' first."
  exit 1
fi
if lsof -i :3000 >/dev/null 2>&1; then
  echo "ERROR: Port 3000 already in use. Run 'bash scripts/stop.sh' first."
  exit 1
fi

# Start backend
echo "Starting FastAPI backend on port 8000..."
python -m backend.api &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
echo "Starting React frontend on port 3000..."
cd "$PROJECT_ROOT/frontend"
npm start &
FRONTEND_PID=$!

echo ""
echo "✓ Services started"
echo "  - Backend API: http://localhost:8000"
echo "  - Frontend UI: http://localhost:3000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
