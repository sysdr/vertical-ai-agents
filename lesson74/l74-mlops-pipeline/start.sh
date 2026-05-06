#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DOCKER_MODE=${1:-""}

release_ports() {
  command -v fuser >/dev/null 2>&1 || return 0
  fuser -k 8074/tcp 2>/dev/null || true
  fuser -k 3074/tcp 2>/dev/null || true
}

echo "=== L74 MLOps Pipeline ==="

if [ "$DOCKER_MODE" = "--docker" ]; then
  echo "Starting with Docker Compose..."
  release_ports
  docker compose down 2>/dev/null || true
  docker compose up --build -d
  echo "Backend:  http://localhost:8074"
  echo "Prometheus: http://localhost:9090"
else
  echo "Starting in venv mode..."
  release_ports
  [ ! -d .venv ] && python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q -r requirements.txt
  mkdir -p data
  uvicorn agent_service.main:app --host 0.0.0.0 --port 8074 --reload &
  BACKEND_PID=$!
  echo $BACKEND_PID > .backend.pid
  echo "Backend PID: $BACKEND_PID"

  cd frontend
  npm install --silent
  npm run dev &
  FRONTEND_PID=$!
  echo $FRONTEND_PID > ../.frontend.pid
  echo "Frontend PID: $FRONTEND_PID"
  cd ..

  echo ""
  echo "✅ Backend:  http://localhost:8074"
  echo "✅ Frontend: http://localhost:3074"
  echo "✅ API docs: http://localhost:8074/docs"
  echo "✅ Metrics:  http://localhost:8074/metrics"
  echo ""
  echo "Press Ctrl+C to stop, or run ./stop.sh"
  wait
fi
