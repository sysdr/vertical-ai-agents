#!/bin/bash
echo "Starting L9 Agent Memory System..."

# Start backend
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
python api.py &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "✅ System running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Dashboard: http://localhost:3000"
echo "API: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

wait
