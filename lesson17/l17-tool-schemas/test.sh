#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running L17 tests..."

cd tests
python3 -m pytest test_schemas.py -v
cd ..

echo "✓ All tests passed"
