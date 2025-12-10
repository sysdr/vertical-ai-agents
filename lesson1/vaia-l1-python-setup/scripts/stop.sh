#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🛑 Stopping VAIA L1 Environment..."

cd "$PROJECT_DIR"

# Find and kill processes on port 8000
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:8000)
    if [ ! -z "$PID" ]; then
        echo "Stopping process on port 8000 (PID: $PID)..."
        kill $PID 2>/dev/null || true
        sleep 1
        # Force kill if still running
        kill -9 $PID 2>/dev/null || true
        echo "✅ Server stopped"
    else
        echo "ℹ️  No process found on port 8000"
    fi
elif command -v fuser &> /dev/null; then
    fuser -k 8000/tcp 2>/dev/null || echo "ℹ️  No process found on port 8000"
else
    echo "⚠️  Could not find lsof or fuser. Please stop the server manually."
fi

deactivate 2>/dev/null || true
echo "✅ Environment stopped"
