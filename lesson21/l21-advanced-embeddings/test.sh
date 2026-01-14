#!/bin/bash
echo "🧪 Running L21 tests..."

cd backend
source venv/bin/activate
pytest tests/ -v
cd ..

echo "✓ Tests complete"
