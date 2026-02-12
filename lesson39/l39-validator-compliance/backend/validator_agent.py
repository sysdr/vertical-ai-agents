import os
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
from compliance_engine import ComplianceEngine, ValidationResult
import google.generativeai as genai
import asyncio

_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)

class Document(BaseModel):
    id: str
    content: str
    source: str
    metadata: Dict = {}

class ValidatorResponse(BaseModel):
    approved: bool
    factual_check_passed: bool
    compliance_check_passed: bool
    risk_score: int
    issues: List[str]
    audit_id: str
    timestamp: datetime

class ValidatorAgent:
    """Combined factual consistency (L38) + compliance (L39) validator"""
    
    def __init__(self, domain: str = "healthcare", redis_url: str = "redis://localhost:6379"):
        self.domain = domain
        _model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.llm = genai.GenerativeModel(_model)
        self.compliance_engine = ComplianceEngine(domain, redis_url=redis_url)
    
    async def initialize(self):
        """Initialize async components"""
        await self.compliance_engine.initialize()
    
    async def validate(self, documents: List[Document], query: str, 
                      context: Dict = {}) -> ValidatorResponse:
        """
        Two-stage validation:
        1. Factual consistency check (L38)
        2. Compliance check (L39)
        """
        
        # Stage 1: Factual consistency (from L38)
        factual_passed, factual_issues = await self._check_factual_consistency(
            documents, query
        )
        
        # Stage 2: Compliance check (L39)
        combined_content = "\n\n".join([doc.content for doc in documents])
        compliance_result = await self.compliance_engine.validate(
            combined_content,
            context
        )
        
        # Combine results
        all_issues = factual_issues + [
            f"Compliance: {v.rationale}" for v in compliance_result.violations
        ]
        
        approved = factual_passed and compliance_result.passed
        
        return ValidatorResponse(
            approved=approved,
            factual_check_passed=factual_passed,
            compliance_check_passed=compliance_result.passed,
            risk_score=compliance_result.risk_score,
            issues=all_issues,
            audit_id=compliance_result.audit_id,
            timestamp=datetime.now()
        )
    
    async def _check_factual_consistency(self, documents: List[Document], 
                                         query: str) -> tuple[bool, List[str]]:
        """Check for contradictions between documents (from L38)"""
        
        if len(documents) < 2:
            return True, []
        
        try:
            doc_summaries = "\n\n".join([
                f"Document {i+1} ({doc.source}):\n{doc.content[:500]}"
                for i, doc in enumerate(documents)
            ])
            
            prompt = f"""
Analyze these retrieved documents for factual contradictions or inconsistencies:

Query: {query}

{doc_summaries}

Identify any contradictions, conflicting facts, or inconsistencies.
Respond in JSON format:
{{
    "has_contradictions": true/false,
    "issues": ["issue1", "issue2", ...]
}}
"""
            
            response = await self.llm.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            import json
            result = json.loads(response.text)
            
            has_contradictions = result.get('has_contradictions', False)
            issues = result.get('issues', [])
            
            return not has_contradictions, issues
            
        except Exception as e:
            print(f"⚠️ Factual consistency check failed: {e}")
            # Fail open - don't block on errors
            return True, [f"Factual check error: {str(e)}"]
    
    async def get_stats(self) -> Dict:
        """Get validation statistics"""
        return await self.compliance_engine.get_violation_stats()
    
    async def get_audit_trail(self, limit: int = 50) -> List[Dict]:
        """Get recent audit trail"""
        return await self.compliance_engine.get_audit_trail(limit)
