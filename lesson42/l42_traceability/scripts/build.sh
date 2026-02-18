#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[build] Installing frontend dependencies..."
cd frontend && npm install --silent && npm run build
echo "[build] Done. Frontend built to frontend/build/"
