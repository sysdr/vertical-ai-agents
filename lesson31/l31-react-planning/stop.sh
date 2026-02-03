#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛑 Stopping L31 ReAct Planning System..."

if command -v docker &> /dev/null && [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || docker-compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
fi

if [ -f "$SCRIPT_DIR/.backend.pid" ]; then
    kill $(cat "$SCRIPT_DIR/.backend.pid") 2>/dev/null || true
    rm -f "$SCRIPT_DIR/.backend.pid"
fi

if [ -f "$SCRIPT_DIR/.frontend.pid" ]; then
    kill $(cat "$SCRIPT_DIR/.frontend.pid") 2>/dev/null || true
    rm -f "$SCRIPT_DIR/.frontend.pid"
fi

# Cleanup any remaining processes
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true

echo "✅ Services stopped"
