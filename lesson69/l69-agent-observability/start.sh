#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "▶ Starting L69 backend on :8069"
mkdir -p data
cd backend
../.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8069 --reload &
BACKEND_PID=$!
echo $BACKEND_PID > ../data/backend.pid
cd ..

echo "▶ Starting L69 frontend on :3069"
cd frontend
BROWSER=none PORT=3069 npm start &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../data/frontend.pid
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Backend  → http://localhost:8069"
echo " Frontend → http://localhost:3069"
echo " Docs     → http://localhost:8069/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
wait
