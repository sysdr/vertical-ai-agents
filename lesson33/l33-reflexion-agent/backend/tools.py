"""
Example tools for Reflexion Agent testing
"""
from backend.react_agent import Tool
import requests
import random

def search_tool_func(query: str) -> str:
    """Simulated web search"""
    # In production, this would call actual search API
    
    # Simulate occasional failures for reflexion testing
    if random.random() < 0.2:  # 20% failure rate
        return "Error: Search API rate limit exceeded. Try again in 60 seconds."
    
    if "ceo anthropic" in query.lower():
        return "Dario Amodei is the CEO of Anthropic. He co-founded the company in 2021."
    elif "stock price" in query.lower():
        if "google" in query.lower() or "alphabet" in query.lower():
            return "Alphabet (GOOGL) current price: $142.35 (simulated data)"
        return "Please specify which company's stock price you want."
    elif "weather" in query.lower():
        return "Current weather: 72°F, partly cloudy (simulated data)"
    else:
        return f"Search results for '{query}': [Simulated results - implement actual search API]"

def calculator_tool_func(expression: str) -> str:
    """Simple calculator"""
    try:
        # Safe eval for basic math (production: use AST parser)
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def wikipedia_tool_func(topic: str) -> str:
    """Simulated Wikipedia lookup"""
    # Simulate network issues occasionally
    if random.random() < 0.15:  # 15% failure rate
        return "Error: Connection timeout. Wikipedia API unreachable."
    
    summaries = {
        "anthropic": "Anthropic is an AI safety company founded in 2021 by former OpenAI members, focused on building reliable, interpretable, and steerable AI systems.",
        "reflexion": "Reflexion is a framework that reinforces language agents through linguistic feedback, enabling them to learn from mistakes.",
        "react": "ReAct (Reasoning and Acting) combines chain-of-thought reasoning with action execution in language models."
    }
    
    topic_lower = topic.lower()
    for key, summary in summaries.items():
        if key in topic_lower:
            return summary
    
    return f"No Wikipedia article found for '{topic}'. Try a different search term."

# Create tool instances
search_tool = Tool(
    name="search",
    func=search_tool_func,
    description="Search the web for information. Usage: search('your query')"
)

calculator_tool = Tool(
    name="calculator",
    func=calculator_tool_func,
    description="Perform mathematical calculations. Usage: calculator('2 + 2')"
)

wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_tool_func,
    description="Look up topics on Wikipedia. Usage: wikipedia('topic name')"
)

DEFAULT_TOOLS = [search_tool, calculator_tool, wikipedia_tool]
