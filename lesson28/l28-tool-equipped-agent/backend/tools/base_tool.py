from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.schema = schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return self.schema
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against schema"""
        # Simple validation - check required fields
        required = self.schema.get("required", [])
        for field in required:
            if field not in params:
                raise ValueError(f"Missing required parameter: {field}")
        return params
