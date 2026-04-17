#!/usr/bin/env bash
set -euo pipefail
echo "[test] Running L64 test suite..."
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements-dev.txt
fi
.venv/bin/pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
