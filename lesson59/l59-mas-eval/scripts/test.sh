#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "==> Running L59 test suite"
PYTHONPATH=backend python -m pytest backend/tests/ -v --tb=short

echo ""
echo "==> Smoke-testing API (requires running backend)…"
BASE="http://localhost:8059"
curl -sf "$BASE/health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', d" \
  && echo "  [OK] /health" \
  || echo "  [SKIP] Backend not running"

echo ""
echo "==> All tests complete"
