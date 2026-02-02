import pytest
from app.services.validators import ToolCallValidator, RAGGroundednessValidator

@pytest.mark.asyncio
async def test_tool_call_validator_success():
    validator = ToolCallValidator()
    response = {'tool_calls': [{'tool': 'get_weather', 'parameters': {'location': 'San Francisco'}}]}
    result = await validator.validate(response, 'get_weather', {'location': 'San Francisco'})
    assert result['success'] is True
    assert result['param_score'] == 1.0

@pytest.mark.asyncio
async def test_rag_groundedness_validator():
    validator = RAGGroundednessValidator()
    result = await validator.validate(
        "Python is a programming language.",
        ["Python is an interpreted, high-level programming language."],
        []
    )
    assert result['score'] > 0.5
