#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L4 Parameter Analysis Platform..."

# Check for duplicate services
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use. Stopping existing backend..."
    pkill -f "uvicorn.*main:app" || pkill -f "python.*main.py" || true
    sleep 2
fi

if lsof -ti:3000 >/dev/null 2>&1; then
    echo "⚠️  Port 3000 is already in use. Stopping existing frontend..."
    pkill -f "react-scripts" || true
    sleep 2
fi

# Check if backend venv exists
if [ ! -f "$SCRIPT_DIR/backend/venv/bin/activate" ]; then
    echo "❌ Backend virtual environment not found. Please run ./build.sh first."
    exit 1
fi

# Start backend
echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend
echo "⏳ Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ >/dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    sleep 1
done

# Check if frontend node_modules exists
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not found. Installing..."
    cd "$SCRIPT_DIR/frontend"
    npm install
    cd "$SCRIPT_DIR"
fi

# Start frontend
echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
BROWSER=none npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo "✅ Application started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📄 Backend logs: $SCRIPT_DIR/backend.log"
echo "📄 Frontend logs: $SCRIPT_DIR/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Save PIDs
echo $BACKEND_PID > "$SCRIPT_DIR/.backend.pid"
echo $FRONTEND_PID > "$SCRIPT_DIR/.frontend.pid"

# Wait for user interrupt
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
