import google.generativeai as genai
from typing import List, Dict
from models.pipeline import StageResult

class SynthesizerAgent:
    """Synthesizes validated results into coherent responses"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def synthesize(self, documents: List[Dict], query: str, validation_scores: Dict, budget_available: int) -> StageResult:
        """Generate response from validated documents"""
        try:
            # Build context from documents
            context = "\n\n".join([
                f"[Source: {doc['source']}]\n{doc['content']}"
                for doc in documents
            ])
            
            prompt = f"""Based on the following verified documents, answer the query.
Include citations to sources.

Query: {query}

Documents:
{context}

Provide a clear, accurate answer with citations in [Source: filename] format."""

            response = self.model.generate_content(prompt)
            
            # Estimate tokens
            tokens_used = len(prompt.split()) + len(response.text.split())
            
            if tokens_used > budget_available:
                return StageResult(
                    success=False,
                    error=f"Response generation would exceed budget: {tokens_used} > {budget_available}",
                    tokens_used=0
                )
            
            return StageResult(
                success=True,
                data={
                    "response": response.text,
                    "citations": [doc["source"] for doc in documents],
                    "confidence": validation_scores.get("overall_confidence", 0.0)
                },
                tokens_used=tokens_used,
                metadata={
                    "agent": "synthesizer",
                    "sources_used": len(documents)
                }
            )
            
        except Exception as e:
            # Fallback mock response when API fails (e.g. expired key) - allows demo to complete
            mock_response = f"""Based on the verified documents provided:

{chr(10).join(f"- [{d.get('source', 'unknown')}]: {d.get('content', '')[:80]}..." for d in documents)}

**Answer to query:** The documents above contain relevant information about "{query}". 
For a full AI-generated response, configure a valid GEMINI_API_KEY.

[API fallback - error: {str(e)[:60]}...]"""
            tokens_used = len(mock_response.split()) + 50
            return StageResult(
                success=True,
                data={
                    "response": mock_response,
                    "citations": [doc.get("source", "") for doc in documents],
                    "confidence": validation_scores.get("overall_confidence", 0.0)
                },
                tokens_used=min(tokens_used, budget_available),
                metadata={"agent": "synthesizer", "sources_used": len(documents), "fallback": True}
            )
