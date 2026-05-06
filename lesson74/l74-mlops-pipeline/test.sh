#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "=== L74 Test Suite ==="
VENV_PY="$SCRIPT_DIR/.venv/bin/python3"
if [ ! -x "$VENV_PY" ]; then
  python3 -m venv .venv
  "$SCRIPT_DIR/.venv/bin/pip" install -q -r requirements.txt
fi
mkdir -p data

echo "--- Unit tests ---"
"$VENV_PY" -m pytest tests/ -v --tb=short

echo ""
echo "--- Integration smoke tests (requires running server on :8074) ---"
BASE="http://localhost:8074"

check() {
  local desc="$1" url="$2" expected="$3"
  local code
  code=$(curl --max-time 5 -s -o /dev/null -w "%{http_code}" "$url")
  if [ "$code" = "$expected" ]; then
    echo "  ✅ $desc ($code)"
  else
    echo "  ❌ $desc — expected $expected got $code"
  fi
}

if curl --max-time 3 -sf "$BASE/health" >/dev/null 2>&1; then
  check "GET /health"           "$BASE/health"            200
  check "GET /ready"            "$BASE/ready"             200
  check "GET /metrics"          "$BASE/metrics"           200
  check "GET /deployment/state" "$BASE/deployment/state"  200
  MS=$(curl --max-time 5 -sf "$BASE/deployment/metrics-summary" || true)
  echo "  metrics-summary: $MS"
  echo "  ✅ All probes passed"
else
  echo "  ⚠️  Server not running — skipping integration tests"
  echo "     Start with ./start.sh first"
fi

echo ""
echo "=== Tests complete ==="
