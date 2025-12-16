#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔨 Building VAIA L5 Model Comparison Platform..."

# Backend
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Creating Python virtual environment..."
    rm -rf venv
    # Try to create venv, if it fails, create without pip and install pip manually
    if ! python3 -m venv venv 2>/dev/null; then
        echo "Creating venv without pip..."
        python3 -m venv venv --without-pip 2>/dev/null || {
            echo "⚠️  venv module not available, trying alternative..."
            python3 -m virtualenv venv 2>/dev/null || {
                echo "❌ Cannot create virtual environment. Please install python3-venv: sudo apt install python3-venv"
                exit 1
            }
        }
        # Install pip in venv
        if [ -f "venv/bin/python" ]; then
            echo "Installing pip in venv..."
            curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python 2>/dev/null || {
                echo "⚠️  Could not install pip in venv, trying system pip..."
            }
        fi
    fi
fi

if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment activation script not found"
    exit 1
fi

echo "Installing backend dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet 2>/dev/null || echo "⚠️  pip upgrade skipped"
pip install -r requirements.txt --quiet || {
    echo "❌ Failed to install requirements"
    deactivate
    exit 1
}
deactivate

# Frontend
cd "$PROJECT_ROOT/frontend"
npm install

echo "✅ Build complete!"
