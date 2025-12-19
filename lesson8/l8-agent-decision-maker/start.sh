#!/bin/bash
set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L8 Agent Decision Maker..."

# Check if backend venv exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Backend venv not found. Please run ./build.sh first"
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Please run ./build.sh first"
    exit 1
fi

# Start backend
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
python main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"
deactivate
cd "$SCRIPT_DIR"

# Wait for backend
echo "Waiting for backend..."
sleep 3

# Start frontend
cd "$SCRIPT_DIR/frontend"
PORT=3000 npm start &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"
cd "$SCRIPT_DIR"

echo ""
echo "✅ Services running:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

wait
