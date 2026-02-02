#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running L29 tests..."
cd "$PROJECT_ROOT/backend"
source venv/bin/activate
pytest tests/ -v --asyncio-mode=auto 2>/dev/null || python -m pytest tests/ -v --asyncio-mode=auto
echo "✓ Tests complete!"
