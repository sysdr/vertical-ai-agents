#!/bin/bash
set -e

echo "Running L33 Reflexion Agent tests..."

# Run pytest
pytest tests/ -v --tb=short

echo "✓ All tests passed"
