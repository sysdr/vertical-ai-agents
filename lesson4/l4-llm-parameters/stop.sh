#!/bin/bash

echo "🛑 Stopping L4 Parameter Analysis Platform..."

if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null || true
    rm .backend.pid
fi

if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null || true
    rm .frontend.pid
fi

# Kill any remaining processes
pkill -f "uvicorn" || true
pkill -f "react-scripts" || true

echo "✅ All services stopped"
