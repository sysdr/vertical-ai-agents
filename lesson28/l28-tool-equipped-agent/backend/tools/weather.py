from tools.base_tool import BaseTool
import aiohttp
import logging

logger = logging.getLogger(__name__)

class WeatherTool(BaseTool):
    """Get current weather for a location"""
    
    def __init__(self):
        schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "default": "metric"
                }
            },
            "required": ["location"]
        }
        
        super().__init__(
            name="weather",
            description="Get current weather information for a location",
            schema=schema
        )
    
    async def execute(self, location: str, units: str = "metric") -> dict:
        """Execute weather lookup (mock implementation)"""
        
        # Mock weather data (in production, call real API)
        weather_data = {
            "tokyo": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
            "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
            "new york": {"temp": 18, "condition": "Sunny", "humidity": 55},
        }
        
        location_lower = location.lower()
        data = weather_data.get(location_lower, {
            "temp": 20, 
            "condition": "Clear", 
            "humidity": 60
        })
        
        logger.info(f"Weather lookup for {location}: {data}")
        
        return {
            "location": location,
            "temperature": data["temp"],
            "condition": data["condition"],
            "humidity": data["humidity"],
            "units": units
        }
