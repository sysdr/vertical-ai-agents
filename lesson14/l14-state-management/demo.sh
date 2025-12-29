#!/bin/bash
# Demo script to test API and verify metrics update

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_URL="http://localhost:8000"
SESSION_ID="demo_session_$(date +%s)"

echo "🧪 Running Demo Test"
echo "Session ID: $SESSION_ID"
echo ""

# Test 1: Send a message
echo "1. Sending test message..."
RESPONSE=$(curl -s -X POST "$API_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"Hello, this is a test message\"}")

echo "Response: $RESPONSE"
echo ""

# Wait a moment for state to be saved
sleep 2

# Test 2: Get state and check metrics
echo "2. Checking state metrics..."
STATE=$(curl -s "$API_URL/api/state/$SESSION_ID")

if [ -z "$STATE" ] || [ "$STATE" == "null" ]; then
    echo "❌ ERROR: State not found!"
    exit 1
fi

# Extract metrics
VERSION=$(echo "$STATE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('version', 0))" 2>/dev/null)
TURNS=$(echo "$STATE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_turns', 0))" 2>/dev/null)
TOKENS=$(echo "$STATE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_tokens', 0))" 2>/dev/null)
STATUS=$(echo "$STATE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('state_status', 'N/A'))" 2>/dev/null)

echo "Metrics:"
echo "  Version: $VERSION"
echo "  Total Turns: $TURNS"
echo "  Total Tokens: $TOKENS"
echo "  Status: $STATUS"
echo ""

# Validate metrics
if [ "$VERSION" == "0" ] || [ -z "$VERSION" ]; then
    echo "❌ ERROR: Version is 0 or not set!"
    exit 1
fi

if [ "$TURNS" == "0" ] || [ -z "$TURNS" ]; then
    echo "⚠️  WARNING: Total turns is 0 (might be expected if API call failed)"
else
    echo "✅ Total turns is updating: $TURNS"
fi

if [ "$STATUS" == "N/A" ] || [ -z "$STATUS" ]; then
    echo "❌ ERROR: Status is not set!"
    exit 1
else
    echo "✅ Status is set: $STATUS"
fi

# Test 3: Send another message to verify increment
echo ""
echo "3. Sending second message to verify metrics increment..."
curl -s -X POST "$API_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"This is the second message\"}" > /dev/null

sleep 2

STATE2=$(curl -s "$API_URL/api/state/$SESSION_ID")
VERSION2=$(echo "$STATE2" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('version', 0))" 2>/dev/null)
TURNS2=$(echo "$STATE2" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_turns', 0))" 2>/dev/null)

echo "After second message:"
echo "  Version: $VERSION2 (was $VERSION)"
echo "  Total Turns: $TURNS2 (was $TURNS)"

if [ "$VERSION2" -gt "$VERSION" ]; then
    echo "✅ Version incremented correctly"
else
    echo "⚠️  WARNING: Version did not increment (might be expected)"
fi

echo ""
echo "✅ Demo test completed!"

