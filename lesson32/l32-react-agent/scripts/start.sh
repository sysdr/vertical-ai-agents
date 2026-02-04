#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting L32 ReAct Agent..."

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "Starting with Docker..."
    docker-compose up -d
    echo "✓ Services started"
    echo "  Backend: http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
else
    echo "Starting without Docker..."
    
    # Start backend
    cd backend
    if [ ! -d "venv" ]; then
        python3 -m venv venv --without-pip 2>/dev/null || python3 -m venv venv
        if [ ! -f "venv/bin/pip" ]; then
            curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python3
        fi
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    python app.py &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo "✓ Services started"
    echo "  Backend PID: $BACKEND_PID"
    echo "  Frontend PID: $FRONTEND_PID"
    echo "  Backend: http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
fi
