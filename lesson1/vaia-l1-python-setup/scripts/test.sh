#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧪 Running tests..."
echo ""

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    PYTHON_CMD="venv/Scripts/python"
else
    source venv/bin/activate
    PYTHON_CMD="venv/bin/python"
fi

# Check if Python executable exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python executable not found at $PYTHON_CMD"
    exit 1
fi

# Run pytest
echo "Running pytest..."
$PYTHON_CMD -m pytest tests/ -v --tb=short
