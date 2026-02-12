import os
import yaml
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import google.generativeai as genai
import redis.asyncio as redis
import json
import hashlib

_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)

class Rule(BaseModel):
    id: str
    domain: str
    description: str
    severity: int  # 0-100
    pattern: Optional[str] = None
    requires_classification: bool = False
    classification_prompt: Optional[str] = None
    action: str  # block, warn, filter
    version: str = "1.0"
    effective_date: str
    
class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    rationale: str
    severity: int
    timestamp: datetime

class ValidationResult(BaseModel):
    passed: bool
    risk_score: int
    violations: List[RuleResult]
    audit_id: str
    timestamp: datetime

class ComplianceEngine:
    def __init__(self, domain: str, redis_url: str = "redis://localhost:6379"):
        self.domain = domain
        self.rules: List[Rule] = []
        _model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.llm = genai.GenerativeModel(_model)
        self.redis_client = None
        self.redis_url = redis_url
        self._load_rules()
    
    async def initialize(self):
        """Initialize async resources"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    
    def _load_rules(self):
        """Load compliance rules from YAML (works from backend/ or Docker /app)"""
        base = Path(__file__).resolve().parent
        rules_path = base / 'rules' / 'compliance_rules.yaml'
        if not rules_path.exists():
            rules_path = base.parent / 'rules' / 'compliance_rules.yaml'
        try:
            with open(rules_path, 'r') as f:
                data = yaml.safe_load(f)
                rules_data = data.get('rules', [])
                
                for rule_data in rules_data:
                    if rule_data.get('domain') == self.domain or rule_data.get('domain') == 'all':
                        self.rules.append(Rule(**rule_data))
                
            print(f"✅ Loaded {len(self.rules)} compliance rules for domain: {self.domain}")
        except FileNotFoundError:
            print("⚠️ No compliance rules file found, using empty ruleset")
            self.rules = []
    
    async def validate(self, content: str, context: Dict) -> ValidationResult:
        """Run compliance validation on content"""
        violations = []
        risk_score = 0
        
        # Evaluate each rule
        for rule in self.rules:
            result = await self._evaluate_rule(content, rule, context)
            
            if not result.passed:
                violations.append(result)
                risk_score += rule.severity
        
        # Determine overall decision
        decision = "reject" if risk_score > 70 else "approve"
        audit_id = await self._audit_log(content, violations, decision, risk_score)
        
        return ValidationResult(
            passed=(decision == "approve"),
            risk_score=risk_score,
            violations=violations,
            audit_id=audit_id,
            timestamp=datetime.now()
        )
    
    async def _evaluate_rule(self, content: str, rule: Rule, context: Dict) -> RuleResult:
        """Evaluate a single compliance rule"""
        
        # Check cache first
        cache_key = self._cache_key(content, rule.id)
        cached = await self._get_cached_result(cache_key)
        if cached:
            return cached
        
        # Deterministic pattern matching
        if rule.pattern:
            if re.search(rule.pattern, content, re.IGNORECASE):
                result = RuleResult(
                    rule_id=rule.id,
                    passed=False,
                    rationale=f"Pattern '{rule.pattern}' detected in content",
                    severity=rule.severity,
                    timestamp=datetime.now()
                )
                await self._cache_result(cache_key, result)
                return result
        
        # LLM-based classification for semantic violations
        if rule.requires_classification:
            try:
                prompt = rule.classification_prompt or f"""
Evaluate if this content violates the following compliance rule:
Rule: {rule.description}
Domain: {rule.domain}

Content to evaluate:
{content[:1000]}

Context: {json.dumps(context)}

Respond with only YES or NO, followed by a brief explanation.
"""
                
                response = await self.llm.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=100
                    )
                )
                
                response_text = response.text.strip()
                
                if "YES" in response_text.upper()[:10]:
                    result = RuleResult(
                        rule_id=rule.id,
                        passed=False,
                        rationale=response_text,
                        severity=rule.severity,
                        timestamp=datetime.now()
                    )
                    await self._cache_result(cache_key, result)
                    return result
                    
            except Exception as e:
                print(f"⚠️ LLM evaluation failed for rule {rule.id}: {e}")
                # Fail open on errors to avoid blocking
                pass
        
        # Rule passed
        result = RuleResult(
            rule_id=rule.id,
            passed=True,
            rationale="No violations detected",
            severity=0,
            timestamp=datetime.now()
        )
        await self._cache_result(cache_key, result)
        return result
    
    def _cache_key(self, content: str, rule_id: str) -> str:
        """Generate cache key for rule evaluation"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"compliance:{self.domain}:{rule_id}:{content_hash}"
    
    async def _get_cached_result(self, key: str) -> Optional[RuleResult]:
        """Retrieve cached rule result"""
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return RuleResult(**data)
        except Exception:
            pass
        return None
    
    async def _cache_result(self, key: str, result: RuleResult):
        """Cache rule evaluation result"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(result.model_dump(), default=str)
            )
        except Exception:
            pass
    
    async def _audit_log(self, content: str, violations: List[RuleResult], 
                         decision: str, risk_score: int) -> str:
        """Write immutable audit trail entry"""
        audit_id = hashlib.sha256(
            f"{datetime.now().isoformat()}{content[:100]}".encode()
        ).hexdigest()[:16]
        
        audit_entry = {
            "audit_id": audit_id,
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "content_snippet": content[:200],
            "decision": decision,
            "risk_score": risk_score,
            "violations": [v.model_dump() for v in violations],
            "rules_evaluated": len(self.rules)
        }
        
        # Store in Redis for quick access
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    f"audit:{self.domain}",
                    json.dumps(audit_entry, default=str)
                )
                # Keep last 1000 entries
                await self.redis_client.ltrim(f"audit:{self.domain}", 0, 999)
            except Exception as e:
                print(f"⚠️ Audit logging failed: {e}")
        
        return audit_id
    
    async def get_audit_trail(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent audit trail entries"""
        if not self.redis_client:
            return []
        
        try:
            entries = await self.redis_client.lrange(
                f"audit:{self.domain}",
                0,
                limit - 1
            )
            return [json.loads(e) for e in entries]
        except Exception:
            return []

    async def get_violation_stats(self) -> Dict:
        """Get aggregate violation statistics"""
        audit_entries = await self.get_audit_trail(limit=500)
        
        if not audit_entries:
            return {
                "total_validations": 0,
                "total_violations": 0,
                "approval_rate": 100.0,
                "avg_risk_score": 0,
                "violations_by_rule": {}
            }
        
        total = len(audit_entries)
        violations = sum(1 for e in audit_entries if e['decision'] == 'reject')
        avg_risk = sum(e['risk_score'] for e in audit_entries) / total
        
        violations_by_rule = {}
        for entry in audit_entries:
            for violation in entry.get('violations', []):
                rule_id = violation['rule_id']
                violations_by_rule[rule_id] = violations_by_rule.get(rule_id, 0) + 1
        
        return {
            "total_validations": total,
            "total_violations": violations,
            "approval_rate": ((total - violations) / total * 100) if total > 0 else 100,
            "avg_risk_score": round(avg_risk, 2),
            "violations_by_rule": violations_by_rule
        }
