#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L21: Advanced Embeddings & Chunking..."

# Check for existing services
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8000 already in use. Stopping existing service..."
    pkill -f "uvicorn app.main:app" || true
    sleep 2
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 3000 already in use. Stopping existing service..."
    pkill -f "react-scripts start" || true
    sleep 2
fi

# Start backend
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend
echo "Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check /tmp/backend.log"
    exit 1
fi

# Start frontend
cd "$SCRIPT_DIR/frontend"
BROWSER=none npm start > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo "✓ Backend running on http://localhost:8000 (PID: $BACKEND_PID)"
echo "✓ Frontend running on http://localhost:3000 (PID: $FRONTEND_PID)"
echo "✓ Backend logs: /tmp/backend.log"
echo "✓ Frontend logs: /tmp/frontend.log"
echo "✓ Press Ctrl+C to stop"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    pkill -f "uvicorn app.main:app" || true
    pkill -f "react-scripts start" || true
    echo "✓ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
