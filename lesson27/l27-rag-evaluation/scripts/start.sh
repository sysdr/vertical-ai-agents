#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting L27 RAG Evaluation System..."

# Check for existing processes
if pgrep -f "uvicorn app.main:app" > /dev/null; then
  echo "⚠️  Backend already running. Run ./scripts/stop.sh first."
  exit 1
fi
if pgrep -f "vite" > /dev/null 2>&1; then
  echo "⚠️  Frontend already running. Run ./scripts/stop.sh first."
  exit 1
fi

# Start backend
cd "$PROJECT_ROOT/backend"
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo "✅ System started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop"

wait
