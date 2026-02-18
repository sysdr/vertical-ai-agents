#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[ -f .backend.pid ] && kill $(cat .backend.pid) 2>/dev/null; rm -f .backend.pid
[ -f .frontend.pid ] && kill $(cat .frontend.pid) 2>/dev/null; rm -f .frontend.pid
echo "[stop] Processes stopped."
