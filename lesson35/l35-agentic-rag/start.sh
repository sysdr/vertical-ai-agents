#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting L35 Agentic RAG System ==="

# Start backend
cd backend
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
  uvicorn main:app --host 0.0.0.0 --port 8000 &
elif [ -d .packages ]; then
  export PYTHONPATH="${SCRIPT_DIR}/backend/.packages:${SCRIPT_DIR}/backend:${PYTHONPATH}"
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
else
  python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
[ -f venv/bin/activate ] && deactivate 2>/dev/null || true
cd ..

# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✓ Backend running on http://localhost:8000"
echo "✓ Frontend running on http://localhost:3000"
echo "✓ PIDs: Backend=$BACKEND_PID Frontend=$FRONTEND_PID"

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

wait
