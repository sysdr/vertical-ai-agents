#!/bin/bash
cd backend
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    USE_VENV=true
else
    USE_VENV=false
    echo "Using system Python (no venv found)"
fi

echo "Testing Conversational Agent..."
python3 - << 'PYTHON'
import asyncio
import sys
sys.path.insert(0, '.')
from conversation_engine import ConversationEngine
import os

async def test():
    api_key = os.getenv('GEMINI_API_KEY', '')
    engine = ConversationEngine(api_key, "test.db")
    await engine.initialize()
    
    # Test conversation creation
    conv_id = await engine.create_conversation("test_user")
    assert conv_id, "Failed to create conversation"
    print("✓ Conversation created")
    
    # Test message processing
    result = await engine.process_message(conv_id, "Hello!")
    assert "response" in result, "No response"
    print("✓ Message processed")
    
    # Test goal setting
    result = await engine.process_message(conv_id, "/goal Learn about AI")
    assert result["active_goals"] == 1, "Goal not set"
    print("✓ Goal setting works")
    
    # Test state persistence
    state = await engine.memory.load_state(conv_id)
    assert len(state.messages) == 4, "Messages not persisted"
    print("✓ State persistence works")
    
    print("\n✓ All tests passed!")
    
    # Cleanup
    if os.path.exists("test.db"):
        os.remove("test.db")

asyncio.run(test())
PYTHON

if [ "$USE_VENV" = true ]; then
    deactivate
fi
