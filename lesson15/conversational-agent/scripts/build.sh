#!/bin/bash
set -e
echo "Building Conversational Agent..."

# Install backend dependencies
cd backend
USE_VENV=false
if [ ! -d "venv" ]; then
    if python3 -m venv venv 2>/dev/null; then
        USE_VENV=true
        echo "Created virtual environment"
    else
        echo "Warning: venv creation failed, using system Python with --user flag"
        USE_VENV=false
    fi
fi

if [ "$USE_VENV" = true ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    pip install -q -r requirements.txt
    deactivate
else
    pip3 install --break-system-packages -q -r requirements.txt || pip3 install --user -q -r requirements.txt
fi
cd ..

# Install frontend dependencies
cd frontend
if [ ! -d "node_modules" ]; then
    npm install --silent
fi
cd ..

echo "✓ Build complete"
