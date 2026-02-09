#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building L36 PlannerAgent..."

# Backend
echo "Installing Python dependencies..."
cd backend
if python3 -m venv venv 2>/dev/null; then
  source venv/bin/activate
  pip install -q -r requirements.txt
  deactivate
elif pip3 install --target .packages -r requirements.txt 2>/dev/null; then
  echo "Installed to .packages (venv unavailable)"
else
  pip3 install --user -r requirements.txt 2>/dev/null || \
  pip3 install --break-system-packages -r requirements.txt
fi
cd ..

# Frontend
echo "Installing Node dependencies..."
cd frontend
npm install --silent
cd ..

echo "Build complete!"
