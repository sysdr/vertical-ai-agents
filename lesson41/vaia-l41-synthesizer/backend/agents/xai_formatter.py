from typing import List, Dict, Any
from backend.models.synthesis_models import ReasoningStep, Citation

class XAIFormatter:
    def format_output(self, response: str, reasoning_chain: List[ReasoningStep],
                     citations: List[Citation], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Format output with appropriate XAI detail level"""
        
        requires_full_xai = user_context.get("requires_full_xai", False)
        
        if requires_full_xai:
            return {
                "response": response,
                "citations": [c.model_dump() for c in citations],
                "reasoning_chain": [step.model_dump() for step in reasoning_chain],
                "confidence_scores": self._compute_confidence(reasoning_chain),
                "xai_level": "full"
            }
        else:
            return {
                "response": response,
                "citations": [c.model_dump() for c in citations],
                "reasoning_summary": self._summarize_reasoning(reasoning_chain),
                "overall_confidence": self._compute_confidence(reasoning_chain)["overall"],
                "xai_level": "standard"
            }
    
    def _compute_confidence(self, reasoning_chain: List[ReasoningStep]) -> Dict[str, float]:
        """Compute confidence scores"""
        if not reasoning_chain:
            return {"overall": 0.5, "step_scores": []}
        
        step_scores = [step.confidence for step in reasoning_chain]
        
        # Harmonic mean for overall confidence
        harmonic_mean = len(step_scores) / sum(1/s if s > 0 else 1 for s in step_scores)
        
        return {
            "overall": round(harmonic_mean, 3),
            "step_scores": [round(s, 3) for s in step_scores],
            "min_confidence": round(min(step_scores), 3),
            "max_confidence": round(max(step_scores), 3)
        }
    
    def _summarize_reasoning(self, reasoning_chain: List[ReasoningStep]) -> str:
        """Create simplified reasoning summary"""
        if not reasoning_chain:
            return "No reasoning steps recorded."
        
        summary_parts = []
        for step in reasoning_chain:
            summary_parts.append(f"{step.step_name}: {step.description}")
        
        return " → ".join(summary_parts)
