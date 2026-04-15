#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for f in .pid_backend .pid_frontend; do
  [ -f "$f" ] && kill "$(cat $f)" 2>/dev/null && rm "$f" && echo "Stopped $(basename $f)"
done
pkill -f "uvicorn backend.api.main" 2>/dev/null || true
pkill -f "react-scripts" 2>/dev/null || true
echo "Done."
