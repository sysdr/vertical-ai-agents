#!/bin/bash
echo "Starting Naïve RAG system..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Using Docker deployment..."
    docker-compose up -d
    echo ""
    echo "✓ Services started"
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    echo "  Docs:     http://localhost:8000/docs"
else
    echo "Docker not found. Using local deployment..."
    
    # Setup Python virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    
    # Install backend dependencies
    pip install -r backend/requirements.txt
    
    # Load environment variables from .env file if it exists
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    
    # Check if API key is set
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "Warning: GEMINI_API_KEY not set. Please create a .env file with your API key."
        echo "See .env.example for reference."
    fi
    
    # Start backend
    cd backend && python main.py &
    BACKEND_PID=$!
    cd ..
    
    # Install and start frontend
    cd frontend
    npm install
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo ""
    echo "✓ Services started"
    echo "  Backend PID:  $BACKEND_PID"
    echo "  Frontend PID: $FRONTEND_PID"
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:3000"
    
    # Save PIDs for stop script
    echo $BACKEND_PID > .backend.pid
    echo $FRONTEND_PID > .frontend.pid
fi
