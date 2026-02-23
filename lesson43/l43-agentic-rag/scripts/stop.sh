#!/usr/bin/env bash
[ -f /tmp/l43_backend.pid ]  && kill $(cat /tmp/l43_backend.pid)  2>/dev/null && rm /tmp/l43_backend.pid  && echo "[L43] Backend stopped"
[ -f /tmp/l43_frontend.pid ] && kill $(cat /tmp/l43_frontend.pid) 2>/dev/null && rm /tmp/l43_frontend.pid && echo "[L43] Frontend stopped"
pkill -f "uvicorn backend.api.app" 2>/dev/null || true
echo "[L43] ✓ All processes stopped"
