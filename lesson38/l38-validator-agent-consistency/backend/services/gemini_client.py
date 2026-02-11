import google.generativeai as genai
from config.settings import settings
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
    async def validate_consistency(self, doc1_content: str, doc2_content: str) -> dict:
        """
        Use Gemini to check consistency between two documents
        """
        prompt = f"""Compare these two documents for factual consistency. Identify any contradictions.

Document 1:
{doc1_content[:1000]}

Document 2:
{doc2_content[:1000]}

Respond ONLY with valid JSON in this exact format:
{{
  "consistency_score": <float 0.0-1.0>,
  "contradictions": [<list of specific contradictions found>],
  "confidence": <float 0.0-1.0>
}}

Rules:
- consistency_score: 1.0 = fully consistent, 0.0 = major contradictions
- contradictions: List specific facts that conflict (empty if none)
- confidence: How confident you are in this assessment
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate structure
            if not all(k in result for k in ["consistency_score", "contradictions", "confidence"]):
                raise ValueError("Missing required fields in response")
                
            return result
            
        except Exception as e:
            logger.error(f"Gemini validation error: {str(e)}")
            # Return neutral score on error
            return {
                "consistency_score": 0.7,
                "contradictions": [],
                "confidence": 0.0
            }

gemini_client = GeminiClient()
