#!/bin/bash
set -e

echo "Starting L6: LLM API Client..."

# Start backend
echo "[1/2] Starting FastAPI backend..."
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
echo "[2/2] Starting React frontend..."
cd frontend
HOST=0.0.0.0 \
PORT=3000 \
WDS_SOCKET_HOST=localhost \
WDS_SOCKET_PORT=3000 \
BROWSER=none \
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Application started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
