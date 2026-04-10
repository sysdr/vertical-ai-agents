#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d "$ROOT/.venv" ] || { echo "[L54] Missing .venv — run scripts/build.sh first"; exit 1; }
[ -f "$ROOT/scripts/stop.sh" ] && bash "$ROOT/scripts/stop.sh" || true
mkdir -p data logs
source "$ROOT/.venv/bin/activate"

echo "[L54] Starting backend on :8054..."
uvicorn backend.main:app --host 0.0.0.0 --port 8054 --reload \
  > "$ROOT/logs/backend.log" 2>&1 &
echo $! > "$ROOT/.backend.pid"

sleep 2

echo "[L54] Starting frontend on :3054..."
cd "$ROOT/frontend"
npm run dev > "$ROOT/logs/frontend.log" 2>&1 &
echo $! > "$ROOT/.frontend.pid"
cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  L54 Agent Personalization           ║"
echo "║  Backend  → http://localhost:8054    ║"
echo "║  Frontend → http://localhost:3054    ║"
echo "║  API Docs → http://localhost:8054/docs ║"
echo "╚══════════════════════════════════════╝"
