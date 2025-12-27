#!/bin/bash
# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Starting VAIA L13: Context Engineering..."

# Check if docker-compose.yml exists with full path
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"

if command -v docker &> /dev/null && [ -f "$DOCKER_COMPOSE_FILE" ]; then
    echo "Starting with Docker..."
    cd "$PROJECT_DIR/docker" || exit 1
    docker-compose up -d
    echo "✓ Services started"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
else
    echo "Starting local services..."
    
    # Start backend with full path validation
    BACKEND_DIR="$PROJECT_DIR/backend"
    if [ ! -d "$BACKEND_DIR" ]; then
        echo "Error: Backend directory not found at $BACKEND_DIR"
        exit 1
    fi
    
    cd "$BACKEND_DIR" || exit 1
    
    # Check if venv exists, create if not
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Check if uvicorn is available
    if ! command -v uvicorn &> /dev/null; then
        echo "Error: uvicorn not found. Installing dependencies..."
        pip install -r requirements.txt
    fi
    
    uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd "$PROJECT_DIR" || exit 1
    
    # Start frontend with full path validation
    FRONTEND_DIR="$PROJECT_DIR/frontend"
    if [ ! -d "$FRONTEND_DIR" ]; then
        echo "Error: Frontend directory not found at $FRONTEND_DIR"
        exit 1
    fi
    
    cd "$FRONTEND_DIR" || exit 1
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    npm start &
    FRONTEND_PID=$!
    cd "$PROJECT_DIR" || exit 1
    
    echo "✓ Services started"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
fi
