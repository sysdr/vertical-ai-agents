import pytest
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import JSONParser, PromptEngine, ParseStrategy

@pytest.mark.asyncio
async def test_direct_parse_success():
    parser = JSONParser()
    valid_json = '{"name": "Alice", "age": 30}'
    
    result = await parser._try_direct_parse(valid_json)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.DIRECT
    assert result.data == {"name": "Alice", "age": 30}

@pytest.mark.asyncio
async def test_direct_parse_failure():
    parser = JSONParser()
    invalid_json = '{name: Alice, age: 30}'  # Missing quotes
    
    result = await parser._try_direct_parse(invalid_json)
    
    assert result.success == False
    assert result.strategy == ParseStrategy.DIRECT
    assert result.error is not None

@pytest.mark.asyncio
async def test_regex_extraction_markdown():
    parser = JSONParser()
    text_with_markdown = '''
Here is the JSON you requested:

```json
{"name": "Bob", "age": 25}
```

Hope this helps!
'''
    
    result = await parser._try_regex_extraction(text_with_markdown)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.REGEX
    assert result.data == {"name": "Bob", "age": 25}

@pytest.mark.asyncio
async def test_regex_extraction_embedded():
    parser = JSONParser()
    text_with_embedded = 'The result is {"status": "success", "count": 42} as you can see.'
    
    result = await parser._try_regex_extraction(text_with_embedded)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.REGEX
    assert "status" in result.data

def test_prompt_construction():
    schema = {"name": "str", "age": "int"}
    instruction = "Generate a user"
    
    prompt = PromptEngine.build_json_prompt(instruction, schema)
    
    assert "Generate a user" in prompt
    assert "JSON" in prompt
    # Check for indented JSON format (as used in the prompt)
    assert json.dumps(schema, indent=2) in prompt or json.dumps(schema) in prompt
    assert "valid json" in prompt.lower()

def test_prompt_construction_with_examples():
    schema = {"status": "str"}
    instruction = "Check status"
    examples = [{"status": "active"}, {"status": "inactive"}]
    
    prompt = PromptEngine.build_json_prompt(instruction, schema, examples)
    
    assert "Example" in prompt
    assert "active" in prompt
    assert "inactive" in prompt

@pytest.mark.asyncio
async def test_parse_with_fallback_success():
    parser = JSONParser()
    valid_json = '{"result": "success"}'
    
    result = await parser.parse_with_fallback(valid_json, max_attempts=1)
    
    assert result.success == True
    assert result.data == {"result": "success"}
    assert result.parse_time_ms >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
