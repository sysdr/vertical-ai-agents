#!/bin/bash
set -e

echo "Running L3 Transformer Tests..."

source venv/bin/activate

cd backend
pytest tests/ -v --tb=short

echo ""
echo "✅ All tests passed!"
