#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Building L27 RAG Evaluation System..."

# Backend
echo "Installing Python dependencies..."
cd "$PROJECT_ROOT/backend"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
npm install

echo "✅ Build complete!"
