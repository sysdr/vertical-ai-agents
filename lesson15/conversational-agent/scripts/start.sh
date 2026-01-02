#!/bin/bash
echo "Starting Conversational Agent..."

# Check for existing services
if pgrep -f "uvicorn api:app" > /dev/null; then
    echo "Warning: Backend service already running. Stopping it first..."
    pkill -f "uvicorn api:app"
    sleep 2
fi

if pgrep -f "react-scripts start" > /dev/null; then
    echo "Warning: Frontend service already running. Stopping it first..."
    pkill -f "react-scripts start"
    sleep 2
fi

# Start backend
cd backend
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    USE_VENV=true
else
    USE_VENV=false
    echo "Using system Python (no venv found)"
fi
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend
cd frontend
PORT=3000 npm start &
FRONTEND_PID=$!
cd ..

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
