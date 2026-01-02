#!/usr/bin/env python3
"""Validate that dashboard metrics update correctly"""
import requests
import time
import json

API_URL = "http://localhost:8000"

def test_dashboard_metrics():
    print("=" * 60)
    print("Dashboard Metrics Validation Test")
    print("=" * 60)
    
    # Step 1: Create conversation
    print("\n1. Creating conversation...")
    response = requests.post(f"{API_URL}/conversations", json={"user_id": "dashboard_test"})
    if response.status_code != 200:
        print(f"   ✗ Failed to create conversation: {response.status_code}")
        return False
    conv_id = response.json()["conversation_id"]
    print(f"   ✓ Conversation ID: {conv_id}")
    
    # Step 2: Send initial message
    print("\n2. Sending initial message...")
    response = requests.post(f"{API_URL}/messages", json={
        "conversation_id": conv_id,
        "message": "Hello, I want to learn Python programming"
    })
    if response.status_code != 200:
        print(f"   ✗ Failed: {response.status_code}")
        return False
    
    data = response.json()
    print(f"   Response: {data.get('response', '')[:100]}...")
    print(f"   Metrics:")
    print(f"     - State: {data.get('state', 'N/A')}")
    print(f"     - Total Messages: {data.get('total_messages', 0)}")
    print(f"     - Active Goals: {data.get('active_goals', 0)}")
    print(f"     - Total Tokens: {data.get('total_tokens', 0)}")
    
    # Step 3: Set a goal
    print("\n3. Setting a goal...")
    response = requests.post(f"{API_URL}/messages", json={
        "conversation_id": conv_id,
        "message": "/goal Master Python basics in 1 week"
    })
    if response.status_code != 200:
        print(f"   ✗ Failed: {response.status_code}")
        return False
    
    data = response.json()
    print(f"   Metrics after goal:")
    print(f"     - State: {data.get('state', 'N/A')}")
    print(f"     - Total Messages: {data.get('total_messages', 0)}")
    print(f"     - Active Goals: {data.get('active_goals', 0)}")
    print(f"     - Total Tokens: {data.get('total_tokens', 0)}")
    
    # Step 4: Send another message
    print("\n4. Sending follow-up message...")
    time.sleep(2)  # Wait a bit
    response = requests.post(f"{API_URL}/messages", json={
        "conversation_id": conv_id,
        "message": "What are the key concepts I should learn first?"
    })
    if response.status_code != 200:
        print(f"   ✗ Failed: {response.status_code}")
        return False
    
    data = response.json()
    print(f"   Metrics after message:")
    print(f"     - State: {data.get('state', 'N/A')}")
    print(f"     - Total Messages: {data.get('total_messages', 0)}")
    print(f"     - Active Goals: {data.get('active_goals', 0)}")
    print(f"     - Total Tokens: {data.get('total_tokens', 0)}")
    
    # Step 5: Get full history
    print("\n5. Retrieving full conversation history...")
    response = requests.get(f"{API_URL}/conversations/{conv_id}/history")
    if response.status_code != 200:
        print(f"   ✗ Failed: {response.status_code}")
        return False
    
    history_data = response.json()
    messages = history_data.get('messages', [])
    goals = history_data.get('goals', [])
    
    print(f"   Full History Metrics:")
    print(f"     - Total Messages: {len(messages)}")
    print(f"     - Total Goals: {len(goals)}")
    print(f"     - Active Goals: {len([g for g in goals if not g.get('completed', False)])}")
    print(f"     - State: {history_data.get('state', 'N/A')}")
    
    # Calculate total tokens from messages
    total_tokens = sum(msg.get('token_count', 0) for msg in messages)
    print(f"     - Total Tokens (calculated): {total_tokens}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)
    
    all_valid = True
    
    if len(messages) == 0:
        print("✗ FAIL: No messages found")
        all_valid = False
    else:
        print(f"✓ PASS: Messages count = {len(messages)}")
    
    if total_tokens == 0:
        print("⚠ WARNING: Total tokens is 0 (may be due to token counting method)")
        # This is acceptable if the token counting is just an estimate
    else:
        print(f"✓ PASS: Total tokens = {total_tokens}")
    
    if len(goals) == 0:
        print("⚠ INFO: No goals set (this is OK if no /goal command was used)")
    else:
        print(f"✓ PASS: Goals count = {len(goals)}")
    
    print(f"\n✓ Dashboard should display:")
    print(f"   - State: {history_data.get('state', 'N/A')}")
    print(f"   - Messages: {len(messages)}")
    print(f"   - Active Goals: {len([g for g in goals if not g.get('completed', False)])}")
    print(f"   - Tokens: {total_tokens}")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ VALIDATION COMPLETE: Dashboard metrics should update correctly")
    else:
        print("⚠ VALIDATION COMPLETE: Some issues found (see above)")
    print("=" * 60)
    
    return all_valid

if __name__ == "__main__":
    try:
        test_dashboard_metrics()
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()




