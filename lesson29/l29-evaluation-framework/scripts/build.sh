#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building L29 Evaluation Framework..."
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT/backend"
if ! [ -d venv ]; then
  (virtualenv venv 2>/dev/null || python3 -m venv venv) || { echo "Install python3-venv: apt install python3.12-venv"; exit 1; }
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
cd ..

cd "$PROJECT_ROOT/frontend"
npm install
cd ..

if [ ! -f "$PROJECT_ROOT/backend/.env" ] && [ -f "$PROJECT_ROOT/backend/.env.example" ]; then
  cp "$PROJECT_ROOT/backend/.env.example" "$PROJECT_ROOT/backend/.env"
  echo "Created backend/.env from .env.example - set GEMINI_API_KEY for full evaluation"
fi

echo "✓ Build complete!"
