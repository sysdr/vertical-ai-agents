#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
echo "Building L38 ValidatorAgent..."

echo "Installing backend dependencies..."
(cd "$PROJECT_ROOT/backend" && python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt && deactivate)

echo "Installing frontend dependencies..."
(cd "$PROJECT_ROOT/frontend" && npm install --silent)

echo "Build complete!"
