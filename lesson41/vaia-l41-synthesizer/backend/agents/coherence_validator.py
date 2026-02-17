import google.generativeai as genai
from typing import List, Dict, Tuple
from backend.models.synthesis_models import Chunk, Citation

class CoherenceValidator:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def validate(self, response: str, citations: List[Citation], 
                context: List[Chunk]) -> Tuple[bool, Dict[str, any]]:
        """Validate response quality and citation accuracy"""
        
        # Check 1: Citation coverage
        context_str = "\n".join([f"{c.source}: {c.content[:200]}..." for c in context])
        
        prompt = f"""Analyze if the response is well-supported by citations and context.

Response:
{response}

Context Available:
{context_str}

Citations Provided: {len(citations)}

Evaluate:
1. Are factual claims properly cited? (Y/N)
2. Are there unsupported claims? (Y/N)
3. Is the response coherent and well-structured? (Y/N)
4. Overall quality score (0-100)

Format:
CITED: [Y/N]
UNSUPPORTED: [Y/N]
COHERENT: [Y/N]
SCORE: [0-100]"""

        result = self.model.generate_content(prompt)
        validation_text = result.text
        
        # Parse validation results
        cited = "Y" in validation_text.split("CITED:")[1].split("\n")[0] if "CITED:" in validation_text else True
        has_unsupported = "Y" in validation_text.split("UNSUPPORTED:")[1].split("\n")[0] if "UNSUPPORTED:" in validation_text else False
        coherent = "Y" in validation_text.split("COHERENT:")[1].split("\n")[0] if "COHERENT:" in validation_text else True
        
        try:
            score = int(validation_text.split("SCORE:")[1].split("\n")[0].strip())
        except:
            score = 75
        
        # Check 2: Citation accuracy
        accurate_citations = sum(1 for cit in citations if any(cit.chunk_id == c.id for c in context))
        citation_accuracy = accurate_citations / len(citations) if citations else 1.0
        
        is_valid = cited and not has_unsupported and coherent and citation_accuracy >= 0.9 and score >= 70
        
        return is_valid, {
            "properly_cited": cited,
            "has_unsupported_claims": has_unsupported,
            "is_coherent": coherent,
            "quality_score": score,
            "citation_accuracy": citation_accuracy,
            "citation_count": len(citations)
        }
