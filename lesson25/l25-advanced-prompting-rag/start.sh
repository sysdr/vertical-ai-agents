#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load API key from .env so backend gets it (no spaces around = in .env)
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  . "$SCRIPT_DIR/.env"
  set +a
  export GEMINI_API_KEY
  export GOOGLE_API_KEY
  export GEMINI_MODEL
fi

# Check for duplicate services
if command -v lsof >/dev/null 2>&1; then
  if lsof -i :8000 -t 2>/dev/null | head -1 | grep -q .; then
    echo "⚠ Port 8000 in use (backend already running?). Stop with: $SCRIPT_DIR/stop.sh"
    exit 1
  fi
  if lsof -i :3000 -t 2>/dev/null | head -1 | grep -q .; then
    echo "⚠ Port 3000 in use (frontend already running?). Stop with: $SCRIPT_DIR/stop.sh"
    exit 1
  fi
else
  (echo >/dev/tcp/localhost/8000) 2>/dev/null && { echo "⚠ Port 8000 in use. Stop with: $SCRIPT_DIR/stop.sh"; exit 1; }
  (echo >/dev/tcp/localhost/3000) 2>/dev/null && { echo "⚠ Port 3000 in use. Stop with: $SCRIPT_DIR/stop.sh"; exit 1; }
fi

echo "Starting L25 Advanced Prompting RAG..."

# Start backend
cd "$SCRIPT_DIR/backend"
[ -f venv/bin/activate ] && source venv/bin/activate
python3 app.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"

# Start frontend
cd "$SCRIPT_DIR/frontend"
if command -v npm &> /dev/null; then
    PORT=3000 npm start &
    FRONTEND_PID=$!
    echo "✓ Frontend started (PID: $FRONTEND_PID)"
    echo ""
    echo "═══════════════════════════════════════"
    echo "   L25 Advanced Prompting RAG Ready"
    echo "═══════════════════════════════════════"
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    echo "  Docs:     http://localhost:8000/docs"
    echo "═══════════════════════════════════════"
else
    echo "✓ Backend running at http://localhost:8000"
fi

# Save PIDs
cd "$SCRIPT_DIR"
echo $BACKEND_PID > .backend.pid
[ ! -z "$FRONTEND_PID" ] && echo $FRONTEND_PID > .frontend.pid

wait
