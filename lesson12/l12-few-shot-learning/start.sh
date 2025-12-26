#!/bin/bash
echo "🚀 Starting Few-Shot Learning System..."

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Start backend
cd backend || exit 1
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    export GEMINI_API_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
elif [ -f "$HOME/.local/bin/uvicorn" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    export GEMINI_API_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
else
    echo "❌ Backend dependencies not found. Please run ./build.sh first."
    exit 1
fi
cd "$SCRIPT_DIR" || exit 1

# Start frontend
cd frontend || exit 1
if [ ! -d "node_modules" ]; then
    echo "❌ Frontend node_modules not found. Please run ./build.sh first."
    exit 1
fi
PORT=3000 npm start &
FRONTEND_PID=$!
cd "$SCRIPT_DIR" || exit 1

echo "✅ Services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

wait
