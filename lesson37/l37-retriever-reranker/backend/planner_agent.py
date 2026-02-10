import os
import google.generativeai as genai
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

class PlannerAgent:
    """
    Query decomposition from L36
    Breaks complex queries into sub-questions
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def decompose(self, query: str) -> List[str]:
        """Decompose query into sub-queries"""
        
        prompt = f"""Break this query into 2-4 focused sub-questions for knowledge retrieval.

Query: {query}

Return ONLY a JSON array: ["sub-question 1", "sub-question 2", ...]
No explanation."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith("```"):
                text = text.split("```")[1].replace("json", "").strip()
            
            sub_queries = json.loads(text)
            
            if not isinstance(sub_queries, list):
                logger.warning("Invalid sub-queries format, using original query")
                return [query]
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]
