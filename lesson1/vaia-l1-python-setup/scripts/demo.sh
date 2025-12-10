#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Run demo
COUNT=${1:-10}
echo "🎬 Running demo with $COUNT operations..."
$PYTHON_CMD src/demo.py $COUNT
