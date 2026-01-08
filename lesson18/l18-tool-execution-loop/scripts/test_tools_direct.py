#!/usr/bin/env python3
"""
Direct tool testing script - tests tools without requiring LLM/API key
This verifies that all registered tools work correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import registry
import json

def test_tool(name, inputs):
    """Test a tool directly"""
    print(f"\n🧪 Testing {name}...")
    print(f"   Inputs: {json.dumps(inputs, indent=2)}")
    result = registry.execute(name, inputs)
    if result["success"]:
        print(f"   ✅ Success!")
        print(f"   Result: {json.dumps(result['result'], indent=2)}")
        print(f"   Latency: {result['latency_ms']}ms")
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    return result["success"]

def main():
    print("=" * 60)
    print("🔧 Direct Tool Testing (No API Key Required)")
    print("=" * 60)
    
    # Test all registered tools
    tests = [
        ("get_weather", {"city": "Tokyo", "unit": "celsius"}),
        ("get_weather", {"city": "New York", "unit": "fahrenheit"}),
        ("calculate_revenue", {"year": 2024, "quarter": 3}),
        ("calculate_revenue", {"year": 2024, "quarter": 4}),
        ("search_products", {"query": "laptop", "max_results": 3}),
        ("search_products", {"query": "mouse"}),
        ("document_search", {"query": "VAIA architecture", "top_k": 2}),
    ]
    
    passed = 0
    failed = 0
    
    for tool_name, inputs in tests:
        if test_tool(tool_name, inputs):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Show final stats
    print("\n📈 Execution Statistics:")
    stats = registry.get_stats()
    print(json.dumps(stats, indent=2))
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())


