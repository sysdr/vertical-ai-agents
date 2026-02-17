#!/bin/bash
set -e

echo "Building VAIA L41 Synthesizer Agent..."

# Backend
echo "Setting up Python environment..."
source venv/bin/activate
pip install -r requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
