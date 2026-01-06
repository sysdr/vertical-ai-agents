#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building L17 Tool Schema Validator..."

# Backend
cd backend
# Try to create venv, if it fails, use system pip
if [ ! -d "venv" ]; then
    if python3 -m venv venv 2>/dev/null && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        pip install -q -r requirements.txt
        deactivate
    else
        echo "Warning: venv creation failed, using system pip with --break-system-packages flag"
        python3 -m pip install --break-system-packages -q -r requirements.txt
    fi
else
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        pip install -q -r requirements.txt
        deactivate
    else
        echo "Warning: venv exists but activate script missing, using system pip"
        python3 -m pip install --break-system-packages -q -r requirements.txt
    fi
fi
cd ..

# Frontend
cd frontend
if [ ! -d "node_modules" ]; then
    npm install --silent
fi
cd ..

echo "✓ Build complete"
