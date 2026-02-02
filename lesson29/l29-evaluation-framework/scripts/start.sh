#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/backend/.env" ]; then
  set -a
  source "$PROJECT_ROOT/backend/.env"
  set +a
fi

# Check for existing processes
if pgrep -f "uvicorn app.main:app" > /dev/null 2>&1; then
  echo "⚠️  Backend already running. Stop with: $SCRIPT_DIR/stop.sh"
fi
if pgrep -f "vite" > /dev/null 2>&1; then
  echo "⚠️  Frontend already running. Stop with: $SCRIPT_DIR/stop.sh"
fi

echo "Starting L29 Evaluation Framework..."
cd "$PROJECT_ROOT/backend"
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo "✓ Services started!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
