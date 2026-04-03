#!/bin/bash
set -e
echo "=== L49 Start ==="
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  if [ "$(docker compose ps --services --filter status=running | awk 'NF{c++} END{print c+0}')" -gt 0 ]; then
    echo "⚠ Services already running via Docker Compose."
    docker compose ps
    exit 0
  fi
  docker compose up -d
  echo "⏳ Waiting for services..."
  sleep 8
  echo "✅ Backend: http://localhost:8049/docs"
  echo "✅ Frontend: http://localhost:3049"
else
  echo "Starting without Docker..."
  if ss -ltn | awk '{print $4}' | awk 'BEGIN{hit=0} /:8049$|:3049$/ {hit=1} END{exit !hit}'; then
    echo "⚠ Port 8049 or 3049 is already in use. Stop existing services first."
    exit 1
  fi
  if [ -f backend.pid ] && kill -0 $(cat backend.pid) 2>/dev/null; then
    echo "⚠ Backend process already running with PID $(cat backend.pid)"
    exit 0
  fi
  if [ -f frontend.pid ] && kill -0 $(cat frontend.pid) 2>/dev/null; then
    echo "⚠ Frontend process already running with PID $(cat frontend.pid)"
    exit 0
  fi
  cd backend
  source .venv/bin/activate
  nohup uvicorn main:app --host 0.0.0.0 --port 8049 --reload > ../backend.log 2>&1 &
  echo $! > ../backend.pid
  deactivate
  cd ../frontend
  nohup npm start > ../frontend.log 2>&1 &
  echo $! > ../frontend.pid
  echo "✅ Backend: http://localhost:8049/docs"
  echo "✅ Frontend: http://localhost:3049"
fi
