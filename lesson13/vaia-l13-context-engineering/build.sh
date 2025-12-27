#!/bin/bash
echo "Building VAIA L13: Context Engineering..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Building with Docker..."
    cd docker
    docker-compose build
    echo "✓ Docker build complete"
else
    echo "Docker not found, setting up local environment..."
    
    # Backend setup
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    cd ..
    
    # Frontend setup
    cd frontend
    npm install
    cd ..
    
    echo "✓ Local build complete"
fi
