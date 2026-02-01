#!/bin/bash

echo "Building L28 Tool-Equipped Agent..."

# Backend
echo "Installing backend dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"
# Use virtualenv if python3-venv not available
if command -v virtualenv &>/dev/null; then
  virtualenv -p python3 venv
elif python3 -m venv venv 2>/dev/null; then
  true
else
  echo "Error: Need virtualenv or python3-venv. Install with: pip install virtualenv"
  exit 1
fi
source venv/bin/activate
pip install -r requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd ../frontend
npm install

echo "Build complete!"
