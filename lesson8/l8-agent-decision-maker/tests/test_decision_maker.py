"""Tests for L8 Decision Maker"""
import pytest
import asyncio
from backend.main import DecisionMaker, ToolRegistry, AgentGoal

@pytest.fixture
def tool_registry():
    return ToolRegistry()

@pytest.fixture
def decision_maker(tool_registry):
    return DecisionMaker(tool_registry)

@pytest.mark.asyncio
async def test_calculator_tool(tool_registry):
    """Test calculator tool execution"""
    tool = tool_registry.get("calculator")
    result = await tool.execute({"operation": "factorial", "operands": [5]})
    assert result["success"] is True
    assert result["result"] == 120

@pytest.mark.asyncio
async def test_plan_generation(decision_maker):
    """Test plan generation for simple goal"""
    plan = await decision_maker.generate_plan(
        "Calculate 5 factorial",
        {}
    )
    assert plan.estimated_steps > 0
    assert plan.confidence > 0

@pytest.mark.asyncio
async def test_decision_maker_execution(decision_maker):
    """Test full decision maker execution"""
    response = await decision_maker.decision_maker(
        goal="Calculate 5 factorial",
        context={},
        max_steps=5,
        timeout_seconds=10
    )
    assert response.success is True
    assert len(response.traces) > 0

@pytest.mark.asyncio
async def test_tool_registry_list(tool_registry):
    """Test tool listing"""
    tools = tool_registry.list_tools()
    assert len(tools) == 4  # calculator, weather, database, search
    tool_names = [t.name for t in tools]
    assert "calculator" in tool_names

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
