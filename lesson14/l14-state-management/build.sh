#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔨 Building L14 State Management..."

# Setup Python virtual environment
if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/backend/venv" --without-pip 2>/dev/null || python3 -m venv "$SCRIPT_DIR/backend/venv"
    
    # Install pip if venv was created without it
    if [ ! -f "$SCRIPT_DIR/backend/venv/bin/pip" ]; then
        echo "Installing pip in virtual environment..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        "$SCRIPT_DIR/backend/venv/bin/python3" /tmp/get-pip.py
        rm -f /tmp/get-pip.py
    fi
fi

source "$SCRIPT_DIR/backend/venv/bin/activate"

# Install Python dependencies
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/backend/requirements.txt"

# Install frontend dependencies
cd "$SCRIPT_DIR/frontend"
npm install
cd "$SCRIPT_DIR"

echo "✅ Build complete!"
