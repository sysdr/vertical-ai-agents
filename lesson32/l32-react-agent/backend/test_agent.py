import os
import pytest
from dotenv import load_dotenv
from agent import ReActAgent
from tools import ToolRegistry, WikipediaTool, StockPriceTool, CalculatorTool

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY not set in .env - skipping agent tests", allow_module_level=True)

def setup_agent():
    """Setup agent with tools"""
    registry = ToolRegistry()
    registry.register(WikipediaTool())
    registry.register(StockPriceTool())
    registry.register(CalculatorTool())
    return ReActAgent(registry, API_KEY, max_iterations=5)

def test_wikipedia_query():
    """Test Wikipedia search"""
    agent = setup_agent()
    result = agent.execute("What is Python programming language?")
    assert result['success'] or result['iterations'] > 0
    assert 'reasoning_trace' in result

def test_stock_price_query():
    """Test stock price lookup"""
    agent = setup_agent()
    result = agent.execute("What is the current price of Google stock?")
    assert 'reasoning_trace' in result
    assert len(result['reasoning_trace']) > 0

def test_calculation():
    """Test calculator tool"""
    agent = setup_agent()
    result = agent.execute("Calculate 125 * 8")
    assert 'reasoning_trace' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
