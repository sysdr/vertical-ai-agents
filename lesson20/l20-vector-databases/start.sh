#!/bin/bash
# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v docker &> /dev/null; then
    echo "🐳 Starting Docker containers..."
    docker-compose up -d
    echo "✅ Services running:"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
else
    echo "🚀 Starting services..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Run ./build.sh first."
        exit 1
    fi
    
    # Start backend
    if [ -f "backend/main.py" ]; then
        cd backend
        uvicorn main:app --reload --port 8000 &
        BACKEND_PID=$!
        cd ..
        echo "✅ Backend started (PID: $BACKEND_PID)"
    else
        echo "❌ Backend main.py not found"
        exit 1
    fi
    
    # Start frontend
    if [ -f "frontend/package.json" ]; then
        cd frontend
        npm run dev &
        FRONTEND_PID=$!
        cd ..
        echo "✅ Frontend started (PID: $FRONTEND_PID)"
    else
        echo "❌ Frontend package.json not found"
        exit 1
    fi
    
    echo "✅ Services started"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
fi
