#!/bin/bash

echo "🧪 Running L7 tests..."

cd backend
source venv/bin/activate

# Run pytest
python -m pytest ../tests/test_parsing.py -v

deactivate
cd ..

echo "✅ Tests complete!"
