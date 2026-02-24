#!/bin/bash
echo "🔨 Building L44 RAG Evaluation..."
set -e

# Python venv
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "✅ Virtual environment created"
fi
source venv/bin/activate
pip install --quiet -r backend/requirements.txt
echo "✅ Python dependencies installed"

# Frontend
cd frontend && npm install --silent && cd ..
echo "✅ Frontend dependencies installed"
echo "✅ Build complete"
