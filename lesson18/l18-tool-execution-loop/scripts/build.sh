#!/bin/bash
set -e

echo "🔨 Building L18 Tool Execution Loop..."

# Backend
echo "📦 Installing backend dependencies..."
cd backend
# Remove existing venv if present
rm -rf venv
# Create venv without pip (since ensurepip may not be available)
python3 -m venv venv --without-pip
# Use venv's python to install pip
venv/bin/python -m ensurepip --upgrade 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Frontend
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
