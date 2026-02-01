from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for managing and executing tools"""
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
    
    def register(self, tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str):
        """Get tool by name"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]
    
    async def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with parameters"""
        tool = self.get_tool(name)
        
        try:
            # Validate parameters
            validated_params = tool.validate_params(params)
            
            # Execute tool
            result = await tool.execute(**validated_params)
            
            return {
                "success": True,
                "data": result,
                "tool": name
            }
            
        except Exception as e:
            logger.error(f"Tool execution error ({name}): {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool": name
            }
