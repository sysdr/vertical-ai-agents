#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting L17 Tool Schema Validator..."

# Check for existing processes
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "Warning: Port 8000 is already in use. Stopping existing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if lsof -ti:3000 > /dev/null 2>&1; then
    echo "Warning: Port 3000 is already in use. Stopping existing process..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start backend
cd "$SCRIPT_DIR/backend"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    PYTHON_CMD=python
else
    echo "Warning: Using system Python (venv not found)"
    PYTHON_CMD=python3
fi
export $(cat .env | xargs)
$PYTHON_CMD main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend
echo "Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: Backend failed to start. Check backend.log for details."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "Error: Frontend dependencies not installed. Run build.sh first."
    exit 1
fi
BROWSER=none npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

# Store PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo ""
echo "=================================================="
echo "L17 Tool Schema Validator Running!"
echo "=================================================="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Metrics:  http://localhost:8000/metrics"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Logs:"
echo "  Backend:  $SCRIPT_DIR/backend.log"
echo "  Frontend: $SCRIPT_DIR/frontend.log"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="

# Wait
wait
