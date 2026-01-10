#!/bin/bash
echo "Running tests..."

source venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$(pwd):$PYTHONPATH"
pytest tests/ -v --tb=short

echo ""
echo "✓ Tests complete"
