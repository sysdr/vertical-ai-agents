#!/bin/bash
# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v docker &> /dev/null; then
    echo "🐳 Building Docker containers..."
    docker-compose build
else
    echo "📦 Installing dependencies..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r backend/requirements.txt
    cd frontend && npm install && cd ..
fi
echo "✅ Build complete"
