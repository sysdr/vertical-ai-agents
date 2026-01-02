import sys
sys.path.insert(0, '../backend')
from models import ConversationStateModel, Message, Goal
from datetime import datetime

def test_message_serialization():
    msg = Message(role="user", content="test")
    data = msg.dict()
    restored = Message.from_dict(data)
    assert restored.role == msg.role
    assert restored.content == msg.content
    print("✓ Message serialization")

def test_state_operations():
    state = ConversationStateModel(user_id="test", conversation_id="123")
    state.add_message("user", "hello", 10)
    state.add_goal("test goal")
    assert len(state.messages) == 1
    assert len(state.active_goals) == 1
    assert state.total_tokens == 10
    print("✓ State operations")

if __name__ == "__main__":
    test_message_serialization()
    test_state_operations()
    print("\n✓ All model tests passed")
