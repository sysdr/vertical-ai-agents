#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting VAIA L41 Synthesizer Agent..."

# Start backend from project root so backend.* imports resolve
echo "Starting backend server..."
source venv/bin/activate
PYTHONPATH="$SCRIPT_DIR" python backend/main.py &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
echo "Starting frontend..."
cd frontend
BROWSER=none npm start &
FRONTEND_PID=$!
cd ..

echo "✅ Services running:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
