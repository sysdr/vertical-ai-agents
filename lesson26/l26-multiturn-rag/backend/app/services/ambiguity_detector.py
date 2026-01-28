import re
import logging
from typing import Tuple, List
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

class AmbiguityDetector:
    """Detects ambiguous queries and generates clarification questions"""
    
    def __init__(self):
        self.threshold = float(os.getenv("AMBIGUITY_THRESHOLD", "0.4"))
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.ambiguous_patterns = [
            r'\b(that|this|it|these|those)\b(?!\s+\w+)',
            r'^(more|tell me more|what about|and|also)',
            r'\b(he|she|they|them)\b',
            r'^(yes|no|maybe|sure|okay)$',
            r'\b(the other|another|previous|before)\b'
        ]
    
    def detect(
        self,
        query: str,
        conversation_context: str,
        retrieval_confidence: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Detect if query is ambiguous
        
        Returns: (is_ambiguous, ambiguity_score)
        """
        # Linguistic pattern matching
        linguistic_score = self._calculate_linguistic_score(query)
        
        # Retrieval confidence (low confidence = likely ambiguous)
        confidence_score = 1 - retrieval_confidence
        
        # Query length (very short queries often ambiguous)
        length_score = 1.0 if len(query.split()) < 3 else 0.0
        
        # Combined scoring
        ambiguity_score = (
            linguistic_score * 0.5 +
            confidence_score * 0.3 +
            length_score * 0.2
        )
        
        is_ambiguous = ambiguity_score > self.threshold
        
        logger.info(
            f"Ambiguity detection: {ambiguity_score:.3f} "
            f"(linguistic={linguistic_score:.3f}, "
            f"confidence={confidence_score:.3f}, "
            f"length={length_score:.3f}) -> {is_ambiguous}"
        )
        
        return is_ambiguous, ambiguity_score
    
    def _calculate_linguistic_score(self, query: str) -> float:
        """Calculate linguistic ambiguity score"""
        query_lower = query.lower().strip()
        
        matches = sum(
            1 for pattern in self.ambiguous_patterns
            if re.search(pattern, query_lower)
        )
        
        return min(matches / len(self.ambiguous_patterns), 1.0)
    
    def generate_clarifications(
        self,
        query: str,
        conversation_context: str
    ) -> List[str]:
        """Generate clarifying questions"""
        prompt = f"""The user asked: "{query}"

Recent conversation context:
{conversation_context}

This query is ambiguous. Generate 2-3 specific clarifying questions to help understand what the user wants. Make questions actionable and specific, not generic.

Format as a numbered list:
1. [question]
2. [question]
3. [question]"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Parse numbered list
            questions = []
            for line in text.split('\n'):
                match = re.match(r'^\d+\.\s*(.+)$', line.strip())
                if match:
                    questions.append(match.group(1))
            
            return questions[:3]  # Max 3 questions
            
        except Exception as e:
            logger.error(f"Failed to generate clarifications: {e}")
            return [
                "Could you provide more details about what you're looking for?",
                "Which specific aspect are you interested in?",
                "Can you rephrase your question with more context?"
            ]

ambiguity_detector = AmbiguityDetector()
