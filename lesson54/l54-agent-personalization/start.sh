#!/usr/bin/env bash
# L54 — start dashboard (backend + Vite). Run from lesson54/: ./start.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$HERE/l54-agent-personalization"

if [ ! -d "$PROJECT" ]; then
  echo "[L54] Project dir missing — running setup.sh (one-time)..."
  bash "$HERE/setup.sh"
fi

if [ ! -d "$PROJECT/.venv" ] || [ ! -d "$PROJECT/frontend/node_modules" ]; then
  echo "[L54] Installing Python/Node dependencies..."
  bash "$PROJECT/scripts/build.sh"
fi

exec bash "$PROJECT/scripts/start.sh"
