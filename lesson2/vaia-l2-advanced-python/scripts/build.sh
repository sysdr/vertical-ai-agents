#!/bin/bash
set -e

echo "=== Building VAIA L2 ==="

# Backend setup (reuse L1's venv concept)
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ../frontend
echo "Installing Node dependencies..."
npm install

echo "✅ Build complete!"
