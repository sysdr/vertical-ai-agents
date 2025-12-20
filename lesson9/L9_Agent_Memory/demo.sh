#!/bin/bash
# Demo script to populate the memory system with sample data
# This ensures the dashboard shows meaningful metrics

API_BASE="http://localhost:8000"

echo "Running demo to populate memory system..."

# Store multiple short-term messages
echo "Storing short-term messages..."
for i in {1..5}; do
    curl -s -X POST "$API_BASE/api/store" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"demo_session_$i\",\"role\":\"user\",\"content\":\"Demo message $i\"}" > /dev/null
    curl -s -X POST "$API_BASE/api/store" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\":\"demo_session_$i\",\"role\":\"assistant\",\"content\":\"Response to message $i\"}" > /dev/null
done

# Store long-term facts
echo "Storing long-term facts..."
curl -s -X POST "$API_BASE/api/longterm/store" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo_user","fact":{"skill":"Python","level":"advanced"}}' > /dev/null
curl -s -X POST "$API_BASE/api/longterm/store" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo_user","fact":{"skill":"JavaScript","level":"intermediate"}}' > /dev/null
curl -s -X POST "$API_BASE/api/longterm/store" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo_user","fact":{"preference":"React","type":"framework"}}' > /dev/null

# Trigger recall operations
echo "Triggering recall operations..."
for i in {1..3}; do
    curl -s "$API_BASE/api/recall/demo_session_$i" > /dev/null
done

# Trigger long-term searches
echo "Triggering long-term searches..."
curl -s -X POST "$API_BASE/api/longterm/search" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo_user","keywords":["Python"],"limit":5}' > /dev/null
curl -s -X POST "$API_BASE/api/longterm/search" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"demo_user","keywords":["JavaScript"],"limit":5}' > /dev/null

echo "✅ Demo data populated!"
echo "Check metrics at: $API_BASE/api/metrics"


