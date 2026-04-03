#!/bin/bash
echo "=== L49 Stop ==="
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  docker compose down
else
  [ -f backend.pid ] && kill $(cat backend.pid) 2>/dev/null || true
  [ -f frontend.pid ] && kill $(cat frontend.pid) 2>/dev/null || true
  rm -f backend.pid frontend.pid
fi
echo "✅ Stopped"
