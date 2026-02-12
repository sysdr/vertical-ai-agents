#!/bin/bash
echo "🔨 Building L39 Validator Compliance System..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "📦 Building with Docker..."
    docker-compose build
    echo "✅ Docker build complete!"
else
    echo "🐍 Building without Docker (local Python)..."
    
    # Backend
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    
    # Frontend
    cd frontend
    npm install
    cd ..
    
    echo "✅ Local build complete!"
fi
