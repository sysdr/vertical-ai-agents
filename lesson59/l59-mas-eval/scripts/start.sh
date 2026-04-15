#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
mkdir -p data logs

echo "==> Starting backend (port 8059)…"
PYTHONPATH=backend uvicorn backend.api.main:app \
  --host 0.0.0.0 --port 8059 --reload &
echo $! > .pid_backend

sleep 2

if command -v node >/dev/null 2>&1 && [ -d "frontend/node_modules" ]; then
  echo "==> Starting frontend (port 3059)…"
  cd frontend && PORT=3059 BROWSER=none npm start &
  echo $! > ../.pid_frontend
  cd ..
fi

echo ""
echo "  Backend  : http://localhost:8059"
echo "  Frontend : http://localhost:3059"
echo "  Health   : http://localhost:8059/health"
echo "  API docs : http://localhost:8059/docs"
echo ""
echo "  Press Ctrl+C to stop"
wait
