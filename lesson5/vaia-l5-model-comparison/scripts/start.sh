#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting VAIA L5 Model Comparison Platform..."

# Check if services are already running
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "⚠️  Backend service already running. Stopping it first..."
    pkill -f "uvicorn app.main:app" || true
    sleep 2
fi

if pgrep -f "react-scripts start" > /dev/null; then
    echo "⚠️  Frontend service already running. Stopping it first..."
    pkill -f "react-scripts start" || true
    sleep 2
fi

# Start backend
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ]; then
    echo "❌ Backend venv not found. Please run build.sh first."
    exit 1
fi
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait a bit for backend to start
sleep 3

# Start frontend
cd "$PROJECT_ROOT/frontend"
BROWSER=none npm start > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo "✅ Services started!"
echo "📊 Backend: http://localhost:8000 (PID: $BACKEND_PID)"
echo "🎨 Frontend: http://localhost:3000 (PID: $FRONTEND_PID)"
echo ""
echo "Logs:"
echo "  Backend: tail -f /tmp/backend.log"
echo "  Frontend: tail -f /tmp/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; pkill -f 'uvicorn app.main:app' 2>/dev/null; pkill -f 'react-scripts start' 2>/dev/null; exit" EXIT INT TERM
wait
