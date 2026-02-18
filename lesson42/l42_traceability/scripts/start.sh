#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
mkdir -p data/traces

# Start backend
echo "[start] Starting FastAPI backend on :8042"
uvicorn backend.api.app:app --host 0.0.0.0 --port 8042 --reload &
BACKEND_PID=$!
echo $BACKEND_PID > .backend.pid

# Start frontend (dev)
echo "[start] Starting React frontend on :3042"
cd frontend
PORT=3042 npm start &
FE_PID=$!
echo $FE_PID > ../.frontend.pid

echo "[start] Backend PID: $BACKEND_PID | Frontend PID: $FE_PID"
echo "[start] Dashboard: http://localhost:3042"
echo "[start] API docs:   http://localhost:8042/docs"
