#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Starting L3 Transformer Visualizer..."

# Check for Docker
if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    docker-compose up -d
    echo ""
    echo "✅ Services started!"
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
else
    echo "Docker not found. Starting manually..."

    # Start backend
    source venv/bin/activate
    cd backend
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..

    # Start frontend
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..

    echo ""
    echo "✅ Services started!"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo ""
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
fi
