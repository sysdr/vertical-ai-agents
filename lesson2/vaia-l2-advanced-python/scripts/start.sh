#!/bin/bash

echo "=== Starting VAIA L2 ==="

# Start backend
cd backend
source venv/bin/activate
python app/main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Start frontend
cd ../frontend
PORT=3000 npm start &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo "✅ System running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Docs:     http://localhost:8000/docs"

# Wait for processes
wait
