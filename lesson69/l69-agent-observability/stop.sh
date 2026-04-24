#!/usr/bin/env bash
cd "$(dirname "$0")"
for f in data/backend.pid data/frontend.pid; do
  [ -f "$f" ] && kill "$(cat $f)" 2>/dev/null && rm "$f" && echo "Stopped $(basename $f .pid)"
done
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true
echo "All services stopped."
