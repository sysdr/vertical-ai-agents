#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Stopping L25 Advanced Prompting RAG..."

if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null
    rm -f .backend.pid
    echo "✓ Backend stopped"
fi

if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null
    rm -f .frontend.pid
    echo "✓ Frontend stopped"
fi

# Cleanup any remaining processes
pkill -f "uvicorn.*app:app" 2>/dev/null
pkill -f "react-scripts start" 2>/dev/null

echo "✓ All services stopped"
