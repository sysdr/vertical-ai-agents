#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
echo "Stopping L38 ValidatorAgent..."
[ -f "$LOGS_DIR/backend.pid" ] && kill $(cat "$LOGS_DIR/backend.pid") 2>/dev/null; rm -f "$LOGS_DIR/backend.pid"
[ -f "$LOGS_DIR/frontend.pid" ] && kill $(cat "$LOGS_DIR/frontend.pid") 2>/dev/null; rm -f "$LOGS_DIR/frontend.pid"
# Fallback: free ports if still in use
command -v fuser >/dev/null 2>&1 && { fuser -k 8000/tcp 2>/dev/null; fuser -k 3000/tcp 2>/dev/null; true; }
echo "All services stopped!"
