import logging
from typing import Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables - try multiple possible locations
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # backend/.env
    Path.cwd() / "backend" / ".env",  # if running from project root
    Path.cwd() / ".env",  # if running from backend directory
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    # Fallback: try loading from current directory
    load_dotenv()

logger = logging.getLogger(__name__)

class ContextExpander:
    """Expands ambiguous queries using conversation context"""
    
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
    
    def expand_query(
        self,
        query: str,
        conversation_context: str
    ) -> str:
        """
        Expand query using conversation history
        
        Returns: Expanded, standalone query
        """
        if not conversation_context.strip():
            return query
        
        prompt = f"""Expand this query into a standalone, specific question using the conversation context:

Recent conversation:
{conversation_context}

Current query: {query}

Expanded query (standalone, specific, preserves user intent):"""
        
        try:
            response = self.model.generate_content(prompt)
            expanded = response.text.strip()
            
            # Clean up common issues
            expanded = expanded.replace('"', '').replace("'", "")
            if expanded.endswith('.'):
                expanded = expanded[:-1]
            
            logger.info(f"Expanded query: '{query}' -> '{expanded}'")
            return expanded
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query expansion failed: {e}")
            # If API key issue, just return original query (better than failing)
            if "API key" in error_msg.lower() or "API_KEY" in error_msg or "expired" in error_msg.lower():
                logger.warning("Gemini API key issue detected in query expansion; using original query")
            return query  # Fallback to original

context_expander = ContextExpander()
