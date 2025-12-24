#!/bin/bash
echo "🚀 Starting L10 Simple Agent..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Using Docker..."
    docker-compose up -d
    echo "✅ Services started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
else
    echo "Using local environment..."
    
    # Start backend
    cd backend
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
    python main.py &
    BACKEND_PID=$!
    
    # Start frontend
    cd ../frontend
    npm start &
    FRONTEND_PID=$!
    
    echo "✅ Services started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
    echo "PIDs: Backend=$BACKEND_PID Frontend=$FRONTEND_PID"
fi
