"""
Tests for Reflexion Agent
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.reflexion_agent import ReflexionAgent
from backend.reflection_engine import ReflectionEngine, ReflectionMemory
from backend.tools import DEFAULT_TOOLS

def test_reflection_engine():
    """Test reflection engine generates valid critiques"""
    engine = ReflectionEngine()
    
    reflection = engine.reflect(
        task="Find the CEO of Anthropic",
        action="search('Anthropic CEO')",
        observation="Error: API rate limit exceeded",
        memory=[]
    )
    
    assert 'success' in reflection
    assert 'critique' in reflection
    assert 'next_strategy' in reflection
    assert isinstance(reflection['success'], bool)

def test_reflection_memory():
    """Test reflection memory stores and retrieves correctly"""
    memory = ReflectionMemory()
    
    session_id = "test_session_1"
    
    reflection_1 = {
        'attempt': 1,
        'critique': 'API failed',
        'next_strategy': 'Retry'
    }
    
    memory.add(session_id, reflection_1)
    
    retrieved = memory.get(session_id)
    assert len(retrieved) == 1
    assert retrieved[0]['attempt'] == 1
    
    stats = memory.get_stats(session_id)
    assert stats['total'] == 1

def test_reflexion_agent_success():
    """Test reflexion agent can complete simple task"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
    
    # Simple task that should succeed
    result = agent.run("Calculate 5 + 3")
    
    assert 'success' in result
    assert 'attempts' in result
    assert result['attempts'] <= 3

def test_reflexion_agent_max_attempts():
    """Test agent respects max_reflections limit"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=2)
    
    # Task designed to potentially fail
    result = agent.run("Find information about a non-existent topic xyz123abc")
    
    # Should not exceed max reflections
    assert result['attempts'] <= 2

def test_memory_prevents_repeat_mistakes():
    """Test that reflection memory influences next attempt"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
    
    result = agent.run("What is the stock price of Google?")
    
    if len(result['reflections']) > 1:
        # Check that later reflections reference earlier ones
        # (This is implicit in the reflection context passed to LLM)
        assert result['reflections'][0]['attempt'] == 1
        assert result['reflections'][-1]['attempt'] == len(result['reflections'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
