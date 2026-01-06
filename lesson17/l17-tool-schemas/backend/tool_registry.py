"""
Production Tool Registry with Pydantic Schema Management
Converts Pydantic models to Gemini AI function declarations
"""
from typing import Type, Callable, Any
from pydantic import BaseModel
import inspect
from schemas import TOOL_SCHEMAS

class ToolRegistry:
    """
    Enterprise-grade tool registry managing validated tools
    Mirrors pattern used by OpenAI Assistants API and Anthropic Claude
    """
    
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self._function_map: dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        function: Callable,
        description: str
    ):
        """Register a tool with validation schemas"""
        self.tools[name] = {
            "name": name,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "description": description,
            "gemini_declaration": self._pydantic_to_gemini(name, input_schema, description)
        }
        self._function_map[name] = function
    
    def _pydantic_to_gemini(
        self, 
        name: str, 
        model: Type[BaseModel], 
        description: str
    ) -> dict:
        """
        Convert Pydantic model to Gemini function declaration
        Critical for type-safe LLM-tool interaction
        """
        schema = model.model_json_schema()
        
        # Extract properties and required fields
        properties = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_type = prop_schema.get("type", "string")
            prop_def = {
                "type": self._json_type_to_gemini(prop_type),
            }
            
            # Add description if available
            if "description" in prop_schema:
                prop_def["description"] = prop_schema["description"]
            
            # Handle enums (Literal types)
            if "enum" in prop_schema:
                prop_def["enum"] = prop_schema["enum"]
            
            # Note: Gemini Schema doesn't support "default" field, so we skip it
            
            properties[prop_name] = prop_def
        
        # Build parameters schema
        # Gemini requires "type": "object" for parameters
        parameters = {
            "type": "object",
            "properties": properties
        }
        
        # Add required fields if any
        required_fields = schema.get("required", [])
        if required_fields:
            parameters["required"] = required_fields
        
        return {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    
    def _json_type_to_gemini(self, json_type: str) -> str:
        """Map JSON Schema types to Gemini types"""
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object"
        }
        return type_map.get(json_type, "string")
    
    def get_gemini_tools(self) -> list[dict]:
        """Get all tools in Gemini function calling format"""
        # Gemini expects tools as a list of dicts with 'function_declarations' key
        return [{
            "function_declarations": [tool["gemini_declaration"] for tool in self.tools.values()]
        }]
    
    def validate_and_execute(self, tool_name: str, parameters: dict) -> dict:
        """
        Validate parameters and execute tool
        Production error handling with detailed feedback
        """
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        input_schema = tool["input_schema"]
        output_schema = tool["output_schema"]
        function = self._function_map[tool_name]
        
        try:
            # Validate input with Pydantic
            validated_input = input_schema(**parameters)
            
            # Execute function
            result = function(validated_input)
            
            # Validate output
            validated_output = output_schema(**result)
            
            return {
                "success": True,
                "tool": tool_name,
                "result": validated_output.model_dump()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "parameters_received": parameters
            }
    
    def get_tool_info(self, tool_name: str) -> dict:
        """Get detailed tool information"""
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        tool = self.tools[tool_name]
        return {
            "name": tool_name,
            "description": tool["description"],
            "input_schema": tool["input_schema"].model_json_schema(),
            "output_schema": tool["output_schema"].model_json_schema(),
            "gemini_declaration": tool["gemini_declaration"]
        }

# Global registry instance
registry = ToolRegistry()
