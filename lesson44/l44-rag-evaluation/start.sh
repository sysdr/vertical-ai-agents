#!/bin/bash
echo "🚀 Starting L44 RAG Evaluation..."
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Avoid duplicate services: stop any existing L44 backend/frontend
if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
  echo "⚠️  Backend already running (uvicorn). Stopping it first..."
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 2
fi
if pgrep -f "react-scripts start" >/dev/null 2>&1; then
  echo "⚠️  Frontend already running. Stopping it first..."
  pkill -f "react-scripts start" 2>/dev/null || true
  sleep 2
fi
# Free ports in case something else is bound
for port in 8044 3000; do
  pid=$(lsof -ti:$port 2>/dev/null) && [ -n "$pid" ] && kill -9 $pid 2>/dev/null && echo "Freed port $port" && sleep 1 || true
done

mkdir -p data/{reports,test-sets}
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export REPORTS_DIR="$PROJECT_ROOT/data/reports"

source venv/bin/activate 2>/dev/null || { echo "Run ./build.sh first"; exit 1; }

# Start backend (run from project root so REPORTS_DIR is correct)
cd "$PROJECT_ROOT/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8044 --reload &
BACKEND_PID=$!
cd "$PROJECT_ROOT"
echo "✅ Backend: http://localhost:8044"
echo "   API docs: http://localhost:8044/docs"

# Wait for backend to be ready before starting frontend (avoids ERR_CONNECTION_REFUSED in browser)
echo "   Waiting for backend to accept connections..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf http://localhost:8044/health >/dev/null 2>&1; then
    echo "   Backend ready."
    break
  fi
  sleep 1
done
if ! curl -sf http://localhost:8044/health >/dev/null 2>&1; then
  echo "⚠️  Backend may not be ready yet; frontend will start anyway. If you see connection errors, wait a few seconds and refresh."
fi

# Start frontend (ensure port 3000 is free)
if command -v lsof >/dev/null 2>&1; then
  pids=$(lsof -ti:3000 2>/dev/null) && [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null && sleep 2 || true
fi
cd "$PROJECT_ROOT/frontend"
BROWSER=none npm start &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"
echo "✅ Frontend: http://localhost:3000"

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║  L44 · RAG Evaluator Running          ║"
echo "║  Dashboard: http://localhost:3000      ║"
echo "║  API:       http://localhost:8044      ║"
echo "║  Docs:      http://localhost:8044/docs ║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop..."
wait $BACKEND_PID $FRONTEND_PID
