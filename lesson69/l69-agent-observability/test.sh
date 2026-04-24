#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "━━━━ L69 Tests ━━━━"
mkdir -p data
cd backend
../.venv/bin/python -m pytest ../tests/ -v --tb=short
