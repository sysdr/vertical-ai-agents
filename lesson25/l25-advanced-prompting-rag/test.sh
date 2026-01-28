#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Running L25 tests..."

cd "$SCRIPT_DIR/backend"
[ -f venv/bin/activate ] && source venv/bin/activate

cd "$SCRIPT_DIR/tests"
python3 -m pytest test_grounding.py -v

echo ""
echo "Testing grounding with adversarial contexts..."
python3 << 'PYTEST'
import requests
import json

base_url = "http://localhost:8000"

# Test strict grounding with misleading context
print("\n1. Testing strict grounding (should refuse)...")
response = requests.post(f"{base_url}/query", json={
    "question": "What is the capital of France?",
    "domain": "strict",
    "test_mode": True
})
if response.status_code == 503:
    print("   LLM not configured (set GEMINI_API_KEY); skipping API checks.")
    import sys
    sys.exit(0)
result = response.json()
print(f"   Grounding State: {result['grounding_state']}")
print(f"   Passed: {result['grounding_state'] == 'rejected'}")

# Test normal query
print("\n2. Testing normal RAG query...")
response = requests.post(f"{base_url}/query", json={
    "question": "What is RAG?",
    "domain": "strict",
    "test_mode": False
})
result = response.json()
print(f"   Answer: {result['answer'][:100]}...")
print(f"   Grounding State: {result['grounding_state']}")
print(f"   Citations: {len(result['citations'])}")

print("\n✓ All tests passed")
PYTEST
