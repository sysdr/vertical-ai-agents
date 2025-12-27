import pytest
from utils.token_counter import TokenCounter

def test_token_counter_basic():
    counter = TokenCounter()
    
    # Test basic counting
    text = "Hello, world!"
    count = counter.count(text)
    assert count > 0
    assert count < 10  # Should be small

def test_token_counter_analysis():
    counter = TokenCounter()
    
    text = "This is a test sentence." * 100
    analysis = counter.analyze(text, max_tokens=1000)
    
    assert "token_count" in analysis
    assert "usage_percent" in analysis
    assert "recommendation" in analysis

@pytest.mark.asyncio
async def test_empty_text():
    counter = TokenCounter()
    count = counter.count("")
    assert count == 0
