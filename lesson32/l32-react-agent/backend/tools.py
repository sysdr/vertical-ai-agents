from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import wikipediaapi
import logging
import re

logger = logging.getLogger(__name__)

class Tool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool action
        Returns: {'success': bool, 'result': str, 'error': Optional[str]}
        """
        pass
    
    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Tool schema for LLM prompt"""
        pass

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, tool: Tool):
        """Register a tool"""
        tool_name = tool.__class__.__name__.replace('Tool', '')
        self.tools[tool_name] = tool
        logger.info(f"Registered tool: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> Dict[str, Tool]:
        """List all registered tools"""
        return self.tools

class WikipediaTool(Tool):
    """Wikipedia search and summary tool"""
    
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='ReActAgent/1.0',
            language='en'
        )
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Search Wikipedia and return summary"""
        try:
            page = self.wiki.page(query)
            if page.exists():
                summary = page.summary[:500]  # Limit to 500 chars
                return {
                    'success': True,
                    'result': f"Wikipedia summary for '{query}': {summary}",
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'result': '',
                    'error': f"No Wikipedia page found for '{query}'"
                }
        except Exception as e:
            return {
                'success': False,
                'result': '',
                'error': str(e)
            }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'description': 'Search Wikipedia and retrieve article summaries',
            'parameters': {
                'query': {'type': 'string', 'description': 'Search term or article title'}
            }
        }

class StockPriceTool(Tool):
    """Stock price lookup tool (simulated for demo)"""
    
    def execute(self, symbol: str) -> Dict[str, Any]:
        """Get stock price for given symbol"""
        try:
            # In production, use real API like Alpha Vantage or Yahoo Finance
            # For demo, return simulated data
            simulated_prices = {
                'GOOGL': 175.32,
                'AAPL': 195.71,
                'MSFT': 425.89,
                'AMZN': 182.45,
                'TSLA': 248.50
            }
            
            symbol = symbol.upper()
            if symbol in simulated_prices:
                price = simulated_prices[symbol]
                return {
                    'success': True,
                    'result': f"Current stock price for {symbol}: ${price:.2f} (simulated data)",
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'result': '',
                    'error': f"Stock symbol '{symbol}' not found in demo database"
                }
        except Exception as e:
            return {
                'success': False,
                'result': '',
                'error': str(e)
            }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'description': 'Get current stock price for a given ticker symbol',
            'parameters': {
                'symbol': {'type': 'string', 'description': 'Stock ticker symbol (e.g., GOOGL, AAPL)'}
            }
        }

class CalculatorTool(Tool):
    """Mathematical calculator tool"""
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """Evaluate mathematical expression"""
        try:
            # Sanitize input - only allow numbers and basic operators
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.\%]+$', expression):
                return {
                    'success': False,
                    'result': '',
                    'error': 'Invalid expression - only numbers and operators allowed'
                }
            
            result = eval(expression)
            return {
                'success': True,
                'result': f"Calculation result: {expression} = {result}",
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': '',
                'error': f"Calculation error: {str(e)}"
            }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'description': 'Perform mathematical calculations',
            'parameters': {
                'expression': {'type': 'string', 'description': 'Mathematical expression (e.g., "2 + 2", "10 * 5")'}
            }
        }

class WeatherTool(Tool):
    """Weather information tool (simulated)"""
    
    def execute(self, location: str) -> Dict[str, Any]:
        """Get weather for location"""
        try:
            # In production, use OpenWeatherMap or similar API
            # For demo, return simulated data
            simulated_weather = {
                'San Francisco': {'temp': 18, 'condition': 'Partly Cloudy'},
                'New York': {'temp': 12, 'condition': 'Clear'},
                'London': {'temp': 10, 'condition': 'Rainy'},
                'Tokyo': {'temp': 22, 'condition': 'Sunny'},
                'Mumbai': {'temp': 30, 'condition': 'Humid'}
            }
            
            location_title = location.title()
            if location_title in simulated_weather:
                weather = simulated_weather[location_title]
                return {
                    'success': True,
                    'result': f"Weather in {location_title}: {weather['temp']}°C, {weather['condition']} (simulated data)",
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'result': '',
                    'error': f"Weather data not available for '{location}' in demo database"
                }
        except Exception as e:
            return {
                'success': False,
                'result': '',
                'error': str(e)
            }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            'description': 'Get current weather information for a location',
            'parameters': {
                'location': {'type': 'string', 'description': 'City name (e.g., San Francisco, London)'}
            }
        }
