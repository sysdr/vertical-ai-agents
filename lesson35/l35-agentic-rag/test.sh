#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Testing L35 Agentic RAG System ==="

cd backend
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
  pytest tests/ -v
  deactivate
elif [ -d .packages ]; then
  export PYTHONPATH="${SCRIPT_DIR}/backend/.packages:${SCRIPT_DIR}/backend:${PYTHONPATH}"
  python3 -m pytest tests/ -v
else
  python3 -m pytest tests/ -v
fi
cd ..

echo "✓ All tests passed"
