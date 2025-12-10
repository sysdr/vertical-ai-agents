#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting VAIA L1 Environment..."
echo "Project directory: $PROJECT_DIR"
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

# Run validation
echo "Running environment validation..."
$PYTHON_CMD src/validate_environment.py

# Check if port is already in use
if command -v lsof &> /dev/null; then
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port 8000 is already in use. Attempting to use existing service..."
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ':8000 '; then
        echo "⚠️  Port 8000 is already in use. Attempting to use existing service..."
    fi
fi

# Start FastAPI server
echo ""
echo "🌐 Starting FastAPI server on http://0.0.0.0:8000"
echo "📊 Dashboard: http://localhost:8000"
echo "🔍 Health: http://localhost:8000/health"
echo "📈 Metrics API: http://localhost:8000/api/metrics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

$PYTHON_CMD -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
