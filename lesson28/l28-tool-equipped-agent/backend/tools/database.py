from tools.base_tool import BaseTool
import logging

logger = logging.getLogger(__name__)

class DatabaseTool(BaseTool):
    """Query mock database"""
    
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query"
                },
                "table": {
                    "type": "string",
                    "enum": ["users", "products", "orders"]
                }
            },
            "required": ["query", "table"]
        }
        
        super().__init__(
            name="database",
            description="Query internal database tables",
            schema=schema
        )
        
        # Mock data
        self.data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "products": [
                {"id": 1, "name": "Widget", "price": 29.99},
                {"id": 2, "name": "Gadget", "price": 49.99}
            ],
            "orders": [
                {"id": 1, "user_id": 1, "product_id": 1, "quantity": 2},
                {"id": 2, "user_id": 2, "product_id": 2, "quantity": 1}
            ]
        }
    
    async def execute(self, query: str, table: str) -> dict:
        """Execute database query"""
        
        if table not in self.data:
            return {"error": f"Table not found: {table}", "success": False}
        
        logger.info(f"Database query: {query} on table {table}")
        
        return {
            "table": table,
            "query": query,
            "results": self.data[table],
            "count": len(self.data[table]),
            "success": True
        }
