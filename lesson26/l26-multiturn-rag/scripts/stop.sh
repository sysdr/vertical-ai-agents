#!/bin/bash
echo "Stopping L26 Multi-Turn RAG System..."

# Stop backend
if [ -f logs/backend.pid ]; then
    kill $(cat logs/backend.pid) 2>/dev/null
    rm logs/backend.pid
    echo "✅ Backend stopped"
fi

# Stop frontend
if [ -f logs/frontend.pid ]; then
    kill $(cat logs/frontend.pid) 2>/dev/null
    rm logs/frontend.pid
    echo "✅ Frontend stopped"
fi

# Kill any remaining processes
pkill -f "uvicorn app.main:app"
pkill -f "react-scripts start"

echo "✅ System stopped"
