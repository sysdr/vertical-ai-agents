#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L7 Prompt Engineering System..."

# Check for duplicate services
echo "Checking for existing services..."
if pgrep -f "uvicorn main:app" > /dev/null; then
    echo "⚠️  Backend service already running. Stopping it first..."
    pkill -f "uvicorn main:app"
    sleep 2
fi

if pgrep -f "react-scripts start" > /dev/null; then
    echo "⚠️  Frontend service already running. Stopping it first..."
    pkill -f "react-scripts start"
    sleep 2
fi

# Start backend
echo "Starting Python backend on http://localhost:8000"
cd "$SCRIPT_DIR/backend"
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found. Run ./build.sh first."
    exit 1
fi
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend
echo "Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Backend failed to start. Check /tmp/backend.log for errors."
    exit 1
fi

# Start frontend
echo "Starting React frontend on http://localhost:3000"
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "⚠️  node_modules not found. Installing dependencies..."
    npm install
fi
npm start > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "✅ System running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo "   Logs: /tmp/backend.log and /tmp/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo ''; echo 'Stopping services...'; pkill -P $BACKEND_PID; pkill -P $FRONTEND_PID; pkill -f 'uvicorn main:app'; pkill -f 'react-scripts start'; echo '✅ Services stopped'; exit" INT TERM
wait
