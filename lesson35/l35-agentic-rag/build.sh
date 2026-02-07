#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building L35 Agentic RAG System ==="

# Backend
cd backend
if python3 -m venv venv 2>/dev/null; then
  source venv/bin/activate
  pip install -r requirements.txt
  deactivate
  rm -rf .packages 2>/dev/null || true
elif pip3 install --target .packages -r requirements.txt 2>/dev/null; then
  echo "Installed to .packages (venv unavailable)"
else
  pip3 install --break-system-packages -r requirements.txt || \
  pip3 install --user --break-system-packages -r requirements.txt
fi
cd ..

# Frontend
cd frontend
npm install
cd ..

echo "✓ Build complete"
