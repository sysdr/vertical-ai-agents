"""
Tool Schema Definitions using Pydantic v2
Production-grade validation for LLM tool calling
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
from datetime import datetime

class WeatherToolInput(BaseModel):
    """Get current weather for a location"""
    location: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="City name or location (e.g., 'Paris', 'New York')"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit"
    )
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate location contains only safe characters"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Location cannot be empty")
        
        # Allow letters, numbers, spaces, commas, hyphens
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'")
        if not all(c in allowed_chars for c in cleaned):
            raise ValueError(f"Location contains invalid characters: {cleaned}")
        
        return cleaned

class WeatherToolOutput(BaseModel):
    """Structured weather response"""
    location: str
    temperature: float
    unit: str
    condition: str
    humidity: int
    timestamp: str

class TimeToolInput(BaseModel):
    """Get current time for a timezone"""
    timezone: str = Field(
        default="UTC",
        description="Timezone (e.g., 'UTC', 'America/New_York', 'Europe/Paris')"
    )
    format: Literal["12h", "24h"] = Field(
        default="24h",
        description="Time format"
    )
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone format"""
        cleaned = v.strip()
        # Basic validation - in production, use pytz.all_timezones
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/+-")
        if not all(c in allowed_chars for c in cleaned):
            raise ValueError(f"Invalid timezone format: {cleaned}")
        return cleaned

class TimeToolOutput(BaseModel):
    """Structured time response"""
    timezone: str
    current_time: str
    format: str
    utc_offset: str

class SearchToolInput(BaseModel):
    """Search for information"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize search query"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty")
        return cleaned

class SearchToolOutput(BaseModel):
    """Structured search response"""
    query: str
    results_count: int
    results: list[dict]
    timestamp: str

# Tool metadata for registration
TOOL_SCHEMAS = {
    "get_weather": {
        "input": WeatherToolInput,
        "output": WeatherToolOutput,
        "description": "Get current weather information for any location worldwide"
    },
    "get_time": {
        "input": TimeToolInput,
        "output": TimeToolOutput,
        "description": "Get current time for any timezone"
    },
    "search": {
        "input": SearchToolInput,
        "output": SearchToolOutput,
        "description": "Search for information on any topic"
    }
}
