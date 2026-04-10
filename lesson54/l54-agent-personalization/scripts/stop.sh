#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
kill_pidfile() {
  local f="$1"
  if [ -f "$f" ]; then
    local pid
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    rm -f "$f"
  fi
}
kill_pidfile "$ROOT/.backend.pid"
kill_pidfile "$ROOT/.frontend.pid"
# Clear duplicate listeners on lesson ports (stale PID files)
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8054/tcp >/dev/null 2>&1 || true
  fuser -k 3054/tcp >/dev/null 2>&1 || true
fi
echo "[L54] Stopped"
