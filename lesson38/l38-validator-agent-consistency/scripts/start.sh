#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"
cd "$PROJECT_ROOT" || exit 1
echo "Starting L38 ValidatorAgent..."
# Stop any existing services to avoid duplicates
bash "$(dirname "$0")/stop.sh" 2>/dev/null || true

if command -v redis-server >/dev/null 2>&1 && ! pgrep -x "redis-server" > /dev/null; then
    redis-server --daemonize yes 2>/dev/null || true
fi
echo "Starting backend..."
if [ -f "$PROJECT_ROOT/backend/venv/bin/activate" ]; then
  (cd "$PROJECT_ROOT/backend" && source venv/bin/activate && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > "$LOGS_DIR/backend.log" 2>&1) &
else
  (cd "$PROJECT_ROOT/backend" && PYTHONPATH=. exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > "$LOGS_DIR/backend.log" 2>&1) &
fi
echo $! > "$LOGS_DIR/backend.pid"
sleep 3
echo "Starting frontend..."
(cd "$PROJECT_ROOT/frontend" && BROWSER=none npm start > "$LOGS_DIR/frontend.log" 2>&1) &
echo $! > "$LOGS_DIR/frontend.pid"
echo "Done. Dashboard: http://localhost:3000"
