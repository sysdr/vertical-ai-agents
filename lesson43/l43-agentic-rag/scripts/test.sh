#!/usr/bin/env bash
set -e
source "/home/systemdr5/git/vertical-ai-agents/lesson43/l43-agentic-rag/.venv/bin/activate"
# GEMINI_API_KEY is used by the backend (already running); do not override if set
export PYTHONPATH="/home/systemdr5/git/vertical-ai-agents/lesson43/l43-agentic-rag"
BASE="http://localhost:8043"

echo "[L43] Running API tests..."

# Health
echo -n "  Health check... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health")
[ "$STATUS" = "200" ] && echo "✓" || { echo "✗ (HTTP $STATUS)"; exit 1; }

# Query (timeout 120s — pipeline may do 429 retries)
echo -n "  Query endpoint... "
RESP=$(curl -s --max-time 120 -X POST "${BASE}/query" -H "Content-Type: application/json" -d '{"query":"What is the incident response protocol?","collection":"enterprise_kb"}')
echo $RESP | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'trace_id' in d and 'response' in d, 'Missing fields'" && echo "✓" || { echo "✗"; exit 1; }

# Audit
TRACE_ID=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['trace_id'])")
echo -n "  Audit retrieval... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/audit/${TRACE_ID}")
[ "$STATUS" = "200" ] && echo "✓" || echo "✗ (HTTP $STATUS)"

# History
echo -n "  History endpoint... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/history?n=5")
[ "$STATUS" = "200" ] && echo "✓" || echo "✗ (HTTP $STATUS)"

echo ""
echo "[L43] ✓ All tests passed!"
echo "      Trace ID: ${TRACE_ID}"
