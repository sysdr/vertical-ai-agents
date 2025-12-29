#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Running L14 State Management Tests..."

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
    echo "❌ Virtual environment not found. Please run ./build.sh first"
    exit 1
fi

cd "$SCRIPT_DIR/backend"
source venv/bin/activate

# Add backend to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"

# Run pytest
pytest tests/ -v

echo "✅ All tests passed!"
