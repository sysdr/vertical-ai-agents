#!/bin/bash
echo "Starting L26 Multi-Turn RAG System..."

# Ensure logs directory exists
mkdir -p logs

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start backend
echo "Starting backend..."
cd backend
# If a previous backend is running, stop it first
if [ -f ../logs/backend.pid ]; then
    if ps -p "$(cat ../logs/backend.pid)" > /dev/null 2>&1; then
        echo "Stopping existing backend process $(cat ../logs/backend.pid)..."
        kill "$(cat ../logs/backend.pid)" 2>/dev/null || true
        sleep 2
    fi
fi
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
echo $! > ../logs/backend.pid
cd ..

# Start frontend
echo "Starting frontend..."
cd frontend
# If a previous frontend is running, stop it first
if [ -f ../logs/frontend.pid ]; then
    if ps -p "$(cat ../logs/frontend.pid)" > /dev/null 2>&1; then
        echo "Stopping existing frontend process $(cat ../logs/frontend.pid)..."
        kill "$(cat ../logs/frontend.pid)" 2>/dev/null || true
        sleep 2
    fi
fi
BROWSER=none npm start > ../logs/frontend.log 2>&1 &
echo $! > ../logs/frontend.pid
cd ..

echo "✅ System started!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Logs:     ./logs/"
