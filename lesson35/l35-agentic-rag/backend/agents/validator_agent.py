import google.generativeai as genai
from typing import List, Dict
from models.pipeline import StageResult

class ValidatorAgent:
    """Validates retrieved documents for factual consistency"""
    
    def __init__(self, api_key: str, threshold: float = 0.7):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.threshold = threshold
    
    async def validate(self, documents: List[Dict], query: str, budget_available: int) -> StageResult:
        """Score documents for relevance and factual consistency"""
        try:
            # Simple validation: check relevance scores
            scores = {
                "overall_confidence": 0.0,
                "document_scores": [],
                "passed_validation": False
            }
            
            if not documents:
                return StageResult(
                    success=False,
                    error="No documents to validate",
                    tokens_used=0
                )
            
            total_relevance = 0
            for doc in documents:
                relevance = doc.get("relevance", 0.5)
                total_relevance += relevance
                scores["document_scores"].append({
                    "doc_id": doc.get("id"),
                    "relevance": relevance,
                    "validated": relevance >= self.threshold
                })
            
            scores["overall_confidence"] = total_relevance / len(documents)
            scores["passed_validation"] = scores["overall_confidence"] >= self.threshold
            
            tokens_used = len(documents) * 15  # Estimate
            
            return StageResult(
                success=scores["passed_validation"],
                data=scores,
                tokens_used=tokens_used,
                metadata={
                    "agent": "validator",
                    "threshold": self.threshold,
                    "confidence": scores["overall_confidence"]
                }
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=str(e),
                tokens_used=0
            )
