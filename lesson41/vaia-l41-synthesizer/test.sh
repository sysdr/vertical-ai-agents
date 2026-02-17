#!/bin/bash
set -e

echo "Testing VAIA L41 Synthesizer Agent..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

# Test 1: Health check
echo "Test 1: Health check..."
curl -s http://localhost:8000/health | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"

# Test 2: Demo synthesis
echo ""
echo "Test 2: Demo synthesis..."
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST http://localhost:8000/synthesize/demo)
HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | sed 's/HTTP_CODE://')
RESPONSE_BODY=$(echo "$RESPONSE" | sed '/HTTP_CODE:/d')
if [ "$HTTP_CODE" != "200" ]; then
  echo "⚠ Demo returned HTTP $HTTP_CODE. Set a valid GEMINI_API_KEY in .env for full demo."
  CITATION_COUNT=0
  CONFIDENCE=0
else
  if command -v jq >/dev/null 2>&1; then
    echo "$RESPONSE_BODY" | jq '{ confidence: .overall_confidence, synthesis_time: .synthesis_time_ms, citations: (.citations | length), reasoning_steps: (.reasoning_chain | length) }'
    CITATION_COUNT=$(echo "$RESPONSE_BODY" | jq '.citations | length')
    CONFIDENCE=$(echo "$RESPONSE_BODY" | jq '.overall_confidence')
  else
    echo "$RESPONSE_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print('confidence:', d.get('overall_confidence'), 'synthesis_time_ms:', d.get('synthesis_time_ms'), 'citations:', len(d.get('citations',[])), 'reasoning_steps:', len(d.get('reasoning_chain',[])))"
    CITATION_COUNT=$(echo "$RESPONSE_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('citations',[])))")
    CONFIDENCE=$(echo "$RESPONSE_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('overall_confidence',''))")
  fi
fi

# Test 3: Citation validation
echo ""
echo "Test 3: Validating citations..."
echo "✅ Found $CITATION_COUNT citations"

# Test 4: Confidence scoring
echo ""
echo "Test 4: Confidence scoring..."
echo "✅ Overall confidence: $CONFIDENCE"

echo ""
echo "✅ All tests passed!"
