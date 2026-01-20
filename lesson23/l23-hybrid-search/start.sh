#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "Starting L23: Hybrid Search System..."

# Check if backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
    echo "❌ Error: Backend directory not found: $BACKEND_DIR"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "❌ Error: Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

# Check if backend venv exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo "❌ Error: Backend virtual environment not found. Please run ./build.sh first."
    exit 1
fi

# Check if backend main.py exists
if [ ! -f "$BACKEND_DIR/main.py" ]; then
    echo "❌ Error: Backend main.py not found: $BACKEND_DIR/main.py"
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "❌ Error: Frontend node_modules not found. Please run ./build.sh first."
    exit 1
fi

# Start backend
cd "$BACKEND_DIR"
source venv/bin/activate
python main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID) at $BACKEND_DIR"
cd "$SCRIPT_DIR"

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "❌ Error: Backend failed to start"
    exit 1
fi

# Start frontend
cd "$FRONTEND_DIR"
BROWSER=none npm start &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID) at $FRONTEND_DIR"
cd "$SCRIPT_DIR"

echo ""
echo "✅ System running!"
echo "📊 Frontend: http://localhost:3000"
echo "🔌 Backend: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop..."

# Keep script running
wait
