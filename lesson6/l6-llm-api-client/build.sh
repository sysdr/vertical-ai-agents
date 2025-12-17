#!/bin/bash
set -e

echo "Building L6: LLM API Client..."

# Backend
echo "[1/2] Setting up Python environment..."
source venv/bin/activate
pip install -r requirements.txt

# Frontend
echo "[2/2] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo "Run './start.sh' to start the application"
