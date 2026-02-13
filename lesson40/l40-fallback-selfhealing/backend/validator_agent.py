import google.generativeai as genai
from models import RetrievalContext, ValidationResult, FailureSignal, FailureType
import random

class EnhancedValidatorAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.compliance_rules = [
            "no_sensitive_data",
            "no_pii_disclosure",
            "factual_accuracy_required"
        ]
    
    async def validate(self, context: RetrievalContext, simulate_failure: bool = False) -> ValidationResult:
        """Validate retrieved content"""
        
        # Simulate validation failure for demonstration
        if simulate_failure or random.random() < 0.3:
            failure_types = [
                FailureType.COMPLIANCE_VIOLATION,
                FailureType.FACTUAL_INCONSISTENCY,
                FailureType.INSUFFICIENT_CONTEXT
            ]
            failure_type = random.choice(failure_types)
            
            return ValidationResult(
                success=False,
                confidence=0.4,
                failure_signal=FailureSignal(
                    type=failure_type,
                    confidence=0.7,
                    details={"reason": f"Simulated {failure_type.value}"},
                    violated_rules=["factual_accuracy_required"] if failure_type == FailureType.FACTUAL_INCONSISTENCY else []
                ),
                message=f"Validation failed: {failure_type.value}"
            )
        
        # Success case
        return ValidationResult(
            success=True,
            confidence=0.9,
            message="Validation passed"
        )
