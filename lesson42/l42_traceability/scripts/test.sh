#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
mkdir -p data/traces
echo "[test] Running L42 test suite..."
python -m pytest tests/ -v --tb=short --asyncio-mode=auto
echo "[test] All tests passed ✓"
