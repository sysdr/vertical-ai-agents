#!/bin/bash
# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "Testing VAIA L13: Context Engineering..."

# Validate backend directory exists
BACKEND_DIR="$PROJECT_DIR/backend"
if [ ! -d "$BACKEND_DIR" ]; then
    echo "Error: Backend directory not found at $BACKEND_DIR"
    exit 1
fi

cd "$BACKEND_DIR" || exit 1

# Try to use venv if it exists, otherwise use system Python with --break-system-packages
USE_VENV=false
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    source venv/bin/activate 2>/dev/null && USE_VENV=true
fi

# Check if pytest is available, install if needed
if ! command -v pytest &> /dev/null; then
    echo "Installing test dependencies..."
    if [ "$USE_VENV" = true ]; then
        pip install -r requirements.txt
    else
        pip install --break-system-packages -r requirements.txt
    fi
fi

# Check if tests directory exists
TESTS_DIR="$BACKEND_DIR/tests"
if [ ! -d "$TESTS_DIR" ]; then
    echo "Error: Tests directory not found at $TESTS_DIR"
    exit 1
fi

# Set PYTHONPATH to include backend directory for imports
export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"

# Run tests
pytest tests/ -v
TEST_EXIT_CODE=$?

cd "$PROJECT_DIR" || exit 1

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✓ Tests complete"
else
    echo "✗ Tests failed with exit code $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi
