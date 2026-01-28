#!/bin/bash
echo "Testing L26 Multi-Turn RAG System..."

cd backend

# Ensure backend package is importable
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Run unit tests
echo "Running unit tests..."
pytest tests/ -v

# Test multi-turn conversation
echo ""
echo "Testing multi-turn conversation..."
python3 << 'PYTHON'
import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

# Check if backend is reachable; if not, skip HTTP test gracefully
try:
    health = requests.get(f"{BASE_URL}/health", timeout=3)
    health.raise_for_status()
except Exception as e:
    print("Backend not reachable at http://localhost:8000; skipping HTTP conversation test.")
    sys.exit(0)

api_base = f"{BASE_URL}/api"

# Create session
resp = requests.post(f"{api_base}/session/new")
resp.raise_for_status()
session_id = resp.json()["session_id"]
print(f"Created session: {session_id}")

# Test conversation flow
queries = [
    "What are your pricing plans?",
    "How much does it cost?",  # Ambiguous
    "The enterprise plan",  # Clarification
    "What about support?",  # Topic shift
]

for query in queries:
    print(f"\n User: {query}")
    resp = requests.post(
        f"{api_base}/query",
        json={"session_id": session_id, "query": query}
    )
    resp.raise_for_status()
    data = resp.json()
    
    if data["is_ambiguous"]:
        print(f"Assistant: [AMBIGUOUS - {data['ambiguity_score']:.2f}]")
        print("Clarifications:", data["clarification_questions"])
    else:
        print(f"Assistant: {data['response']}")
        print(f"Retrieved: {len(data['retrieved_documents'])} docs")

print("\n✅ Tests complete!")
PYTHON

cd ..
