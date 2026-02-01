import asyncio
import google.generativeai as genai
import os
import json
import logging
from llm_client import generate_with_retry_async

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """RAG evaluation using metrics from L27"""
    
    def __init__(self):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.model = genai.GenerativeModel(model_name)
    
    async def evaluate(
        self, 
        query: str,
        response: str,
        context: str,
        sources: list
    ) -> dict:
        """Evaluate RAG response on key metrics (groundedness and relevance run in parallel)"""
        
        # Run groundedness and relevance in parallel to cut evaluation time ~50%
        groundedness, relevance = await asyncio.gather(
            self._evaluate_groundedness(response, context),
            self._evaluate_relevance(query, response)
        )
        
        # Completeness (simple heuristic, no LLM)
        completeness = min(len(response) / 200, 1.0)  # Prefer 200+ char responses
        
        return {
            "groundedness_score": groundedness,
            "relevance_score": relevance,
            "completeness_score": completeness,
            "overall_score": (groundedness + relevance + completeness) / 3,
            "sources_used": len(sources)
        }
    
    async def _evaluate_groundedness(self, response: str, context: str) -> float:
        """Evaluate if response is grounded in context"""
        
        prompt = f"""Rate how well this response is grounded in the provided context.

Context:
{context}

Response:
{response}

Output a JSON with:
- score: 0.0 to 1.0 (1.0 = fully grounded, 0.0 = not grounded)
- reasoning: brief explanation"""

        config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=256)
        text = await generate_with_retry_async(prompt, config=config)
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            return float(data.get("score", 0.5))
        except:
            return 0.5
    
    async def _evaluate_relevance(self, query: str, response: str) -> float:
        """Evaluate response relevance to query"""
        
        prompt = f"""Rate how relevant this response is to the query.

Query: {query}

Response: {response}

Output JSON with score (0.0-1.0) and reasoning."""

        config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=256)
        text = await generate_with_retry_async(prompt, config=config)
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            return float(data.get("score", 0.5))
        except:
            return 0.5
