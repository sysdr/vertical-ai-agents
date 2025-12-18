#!/usr/bin/env python3
import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("Testing API endpoints...")

# Test health
print("\n1. Testing /health...")
response = requests.get(f"{BASE_URL}/health", timeout=5)
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test metrics (before)
print("\n2. Getting initial metrics...")
response = requests.get(f"{BASE_URL}/metrics", timeout=5)
metrics_before = response.json()
print(f"   Total requests: {metrics_before['total_requests']}")

# Test generate endpoint
print("\n3. Testing /generate endpoint...")
payload = {
    "instruction": "Generate a simple user profile for a software engineer",
    "schema": {
        "name": "str",
        "age": "int",
        "email": "str"
    },
    "temperature": 0.1
}

try:
    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        timeout=60
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Parse success: {result['parse_result']['success']}")
        print(f"   Strategy: {result['parse_result']['strategy']}")
        if result.get('parsed_data'):
            print(f"   Parsed data: {json.dumps(result['parsed_data'], indent=2)}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Wait a bit
time.sleep(2)

# Test metrics (after)
print("\n4. Getting metrics after request...")
response = requests.get(f"{BASE_URL}/metrics", timeout=5)
metrics_after = response.json()
print(f"   Total requests: {metrics_after['total_requests']}")
print(f"   Successful parses: {metrics_after['successful_parses']}")
print(f"   Failed parses: {metrics_after['failed_parses']}")
print(f"   Strategy counts: {json.dumps(metrics_after['strategy_counts'], indent=2)}")
print(f"   Success rate: {metrics_after['success_rate']:.2f}%")

print("\n✅ API test complete!")

