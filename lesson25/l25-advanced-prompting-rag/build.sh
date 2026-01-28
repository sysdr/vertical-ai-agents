#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Building L25 Advanced Prompting RAG..."

# Backend
echo "Installing backend dependencies..."
cd backend
if [ ! -d venv ]; then
  python3 -m venv venv 2>/dev/null || { python3 -m venv --without-pip venv 2>/dev/null && curl -sS https://bootstrap.pypa.io/get-pip.py | venv/bin/python3; }
fi
[ -f venv/bin/activate ] && source venv/bin/activate || { echo "⚠ venv not found, using python3"; }
python3 -m pip install -q --upgrade pip 2>/dev/null || pip install -q --upgrade pip 2>/dev/null || true
python3 -m pip install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
if command -v npm &> /dev/null; then
    npm install --silent
else
    echo "⚠ npm not found, skipping frontend build"
fi

cd "$SCRIPT_DIR"
echo "✓ Build complete"
