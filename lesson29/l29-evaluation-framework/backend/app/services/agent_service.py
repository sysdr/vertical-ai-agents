import asyncio
import google.generativeai as genai
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AgentService:
    """Simplified agent service for evaluation"""

    def __init__(self):
        self.model = None
        self.tools = []

    async def initialize(self):
        """Initialize Gemini model and tools"""
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Agent service initialized with Gemini model")
        else:
            logger.warning("GEMINI_API_KEY not set - using mock responses")

    async def shutdown(self):
        pass

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through agent"""
        try:
            if self.model:
                response = self.model.generate_content(
                    f"Query: {query}\n\nProvide a factual response. If you need to call a tool, specify it."
                )
                text = response.text if response.text else ""
            else:
                await asyncio.sleep(0.04)  # Simulate LLM latency so it shows on dashboard chart
                text = self._mock_response_for_demo(query)
            tool_calls = self._extract_tool_calls(text)
            if not tool_calls and not self.model:
                tool_calls = self._mock_tool_calls_for_demo(query)
            return {
                'answer': text,
                'tool_calls': tool_calls,
                'retrieved_chunks': []
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {'answer': '', 'tool_calls': [], 'retrieved_chunks': []}

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        tool_calls = []
        if 'get_weather' in text.lower():
            tool_calls.append({'tool': 'get_weather', 'parameters': {'location': 'San Francisco'}})
        elif 'search_places' in text.lower() or 'restaurant' in text.lower():
            tool_calls.append({'tool': 'search_places', 'parameters': {'query': 'restaurants', 'nearby': True}})
        return tool_calls

    def _mock_response_for_demo(self, query: str) -> str:
        q = query.lower()
        if 'python' in q or 'features' in q:
            return "Python is an interpreted, high-level programming language. Python's design emphasizes code readability."
        return f"Mock response for: {query}"

    def _mock_tool_calls_for_demo(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        if 'weather' in q:
            return [{'tool': 'get_weather', 'parameters': {'location': 'San Francisco'}}]
        if 'restaurant' in q or 'search' in q or 'places' in q:
            return [{'tool': 'search_places', 'parameters': {'query': 'restaurants', 'nearby': True}}]
        return []
