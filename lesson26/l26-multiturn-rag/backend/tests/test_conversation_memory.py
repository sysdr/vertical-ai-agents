import pytest
from app.services.conversation_memory import ConversationMemory

def test_sliding_window():
    memory = ConversationMemory(max_messages=3, strategy="sliding")
    
    for i in range(5):
        memory.add_message("user", f"Message {i}")
    
    messages = memory.get_messages()
    assert len(messages) == 3
    assert messages[0]["content"] == "Message 2"
    assert messages[-1]["content"] == "Message 4"

def test_token_based_pruning():
    memory = ConversationMemory(max_tokens=50, strategy="token-based")
    
    memory.add_message("user", "Short message")
    memory.add_message("assistant", "Another short one")
    memory.add_message("user", " ".join(["word"] * 50))  # Long message
    
    messages = memory.get_messages()
    assert len(messages) >= 1  # Should keep at least the long message
    assert memory._count_tokens() <= 100  # Should be pruned

def test_serialization():
    memory = ConversationMemory(max_messages=5)
    memory.add_message("user", "Test message")
    
    data = memory.to_dict()
    restored = ConversationMemory.from_dict(data)
    
    assert len(restored.get_messages()) == 1
    assert restored.get_messages()[0]["content"] == "Test message"
