#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "=== Stopping L74 ==="
[ -f .backend.pid ]  && kill "$(cat .backend.pid)"  2>/dev/null && rm .backend.pid || true
[ -f .frontend.pid ] && kill "$(cat .frontend.pid)" 2>/dev/null && rm .frontend.pid || true
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8074/tcp 2>/dev/null || true
  fuser -k 3074/tcp 2>/dev/null || true
fi
docker compose down 2>/dev/null || true
echo "Stopped."
