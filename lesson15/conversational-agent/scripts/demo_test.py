#!/usr/bin/env python3
import requests
import time
import json

API_URL = "http://localhost:8000"

print("Testing API and generating metrics...")

# Create conversation
print("\n1. Creating conversation...")
response = requests.post(f"{API_URL}/conversations", json={"user_id": "demo_user"})
if response.status_code == 200:
    conv_id = response.json()["conversation_id"]
    print(f"   ✓ Conversation created: {conv_id}")
else:
    print(f"   ✗ Failed: {response.status_code}")
    exit(1)

# Send messages to generate metrics
messages = [
    "Hello! I'm interested in learning about AI.",
    "/goal Understand machine learning basics",
    "Can you explain what machine learning is?",
    "What are the main types of machine learning?",
    "Thank you, that was helpful!"
]

for i, msg in enumerate(messages, 1):
    print(f"\n{i+1}. Sending message: {msg[:50]}...")
    response = requests.post(f"{API_URL}/messages", json={
        "conversation_id": conv_id,
        "message": msg
    })
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Response received")
        print(f"   State: {data.get('state', 'N/A')}")
        print(f"   Messages: {data.get('total_messages', 0)}")
        print(f"   Active Goals: {data.get('active_goals', 0)}")
        print(f"   Total Tokens: {data.get('total_tokens', 0)}")
    else:
        print(f"   ✗ Failed: {response.status_code} - {response.text}")
    time.sleep(1)

# Get final history
print("\n6. Getting conversation history...")
response = requests.get(f"{API_URL}/conversations/{conv_id}/history")
if response.status_code == 200:
    data = response.json()
    print(f"   ✓ History retrieved")
    print(f"   Total messages: {len(data.get('messages', []))}")
    print(f"   Total goals: {len(data.get('goals', []))}")
    print(f"   State: {data.get('state', 'N/A')}")
    
    # Check metrics are non-zero
    messages = data.get('messages', [])
    total_tokens = sum(msg.get('token_count', 0) for msg in messages)
    print(f"\n   Metrics Summary:")
    print(f"   - Messages: {len(messages)}")
    print(f"   - Goals: {len(data.get('goals', []))}")
    print(f"   - Total Tokens: {total_tokens}")
    
    if len(messages) > 0 and total_tokens > 0:
        print("\n✓ All metrics are non-zero!")
    else:
        print("\n✗ Warning: Some metrics are zero")
else:
    print(f"   ✗ Failed: {response.status_code}")

print("\n✓ Demo execution complete!")




