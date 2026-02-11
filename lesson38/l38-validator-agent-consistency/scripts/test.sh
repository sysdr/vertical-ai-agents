#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
echo "Testing L38 ValidatorAgent..."
if [ -f "$PROJECT_ROOT/backend/venv/bin/activate" ]; then
  (cd "$PROJECT_ROOT/backend" && source venv/bin/activate && pytest tests/ -v)
else
  (cd "$PROJECT_ROOT/backend" && PYTHONPATH=. python3 -m pytest tests/ -v)
fi
