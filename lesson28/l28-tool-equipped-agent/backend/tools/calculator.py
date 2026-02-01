from tools.base_tool import BaseTool
import math
import logging

logger = logging.getLogger(__name__)

class CalculatorTool(BaseTool):
    """Perform mathematical calculations"""
    
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
        
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions safely",
            schema=schema
        )
    
    async def execute(self, expression: str) -> dict:
        """Execute calculation"""
        
        try:
            # Safe eval with math functions
            allowed_names = {
                k: v for k, v in math.__dict__.items() 
                if not k.startswith("__")
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            logger.info(f"Calculation: {expression} = {result}")
            
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
