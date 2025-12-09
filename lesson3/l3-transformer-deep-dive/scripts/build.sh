#!/bin/bash
set -e

echo "Building L3 Transformer Deep Dive..."

# Backend
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "Build complete!"
