#!/usr/bin/env bash
set -euo pipefail
echo "==> Building L59 MAS Eval & Debug"
cd "$(dirname "$0")/.."

# Python venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r backend/requirements.txt -q
echo "  [OK] Python dependencies installed"

# React (optional)
if command -v node >/dev/null 2>&1; then
  cd frontend && npm install -s && cd ..
  echo "  [OK] Node dependencies installed"
else
  echo "  [WARN] node not found — skipping React build"
fi

echo "==> Build complete"
