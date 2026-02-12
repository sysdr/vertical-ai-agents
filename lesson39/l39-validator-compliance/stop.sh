#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛑 Stopping L39 Validator Compliance System..."

if command -v docker &> /dev/null && [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "🐳 Stopping Docker containers..."
    docker-compose down
else
    echo "🐍 Stopping local processes..."
    
    if [ -f "$SCRIPT_DIR/.backend.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.backend.pid") 2>/dev/null || true
        rm -f "$SCRIPT_DIR/.backend.pid"
    fi
    # Fallback: kill by port 8000
    if command -v lsof &>/dev/null; then
        PIDS=$(lsof -i :8000 -sTCP:LISTEN -t 2>/dev/null); [ -n "$PIDS" ] && kill $PIDS 2>/dev/null || true
    fi
    
    if [ -f "$SCRIPT_DIR/.frontend.pid" ]; then
        kill $(cat "$SCRIPT_DIR/.frontend.pid") 2>/dev/null || true
        rm -f "$SCRIPT_DIR/.frontend.pid"
    fi
    if command -v lsof &>/dev/null; then
        PIDS=$(lsof -i :3000 -sTCP:LISTEN -t 2>/dev/null); [ -n "$PIDS" ] && kill $PIDS 2>/dev/null || true
    fi
    
    # Stop Redis
    redis-cli shutdown 2>/dev/null || true
fi

echo "✅ System stopped!"
