#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "Building L23: Hybrid Search System..."

# Backend
echo "Installing Python dependencies..."
cd "$BACKEND_DIR"

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation..."
else
    # Try to create venv, if it fails, try alternative methods
    if ! python3 -m venv venv 2>/dev/null; then
        echo "Warning: python3 -m venv failed, trying python3 -m virtualenv..."
        if command -v virtualenv &> /dev/null; then
            python3 -m virtualenv venv
        else
            echo "Installing python3-venv..."
            sudo apt-get update && sudo apt-get install -y python3.12-venv || {
                echo "Failed to install python3-venv. Trying alternative approach..."
                python3 -m ensurepip --default-pip || true
                python3 -m venv venv || {
                    echo "❌ Error: Could not create virtual environment"
                    exit 1
                }
            }
        fi
    fi
fi

source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt
deactivate
cd "$SCRIPT_DIR"

# Frontend
echo "Installing Node dependencies..."
cd "$FRONTEND_DIR"
npm install
cd "$SCRIPT_DIR"

echo "✅ Build complete!"
