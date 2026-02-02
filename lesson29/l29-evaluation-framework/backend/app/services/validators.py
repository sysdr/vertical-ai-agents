import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ToolCallValidator:
    async def validate(self, response: Dict[str, Any], expected_tool: str,
                      expected_params: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls = response.get('tool_calls', [])
        if not tool_calls:
            return {'success': False, 'actual_tool': None, 'param_score': 0.0}
        actual = tool_calls[0]
        actual_tool = actual.get('tool', '')
        actual_params = actual.get('parameters', {})
        tool_match = actual_tool == expected_tool
        param_score = self._fuzzy_match_params(actual_params, expected_params)
        success = tool_match and param_score >= 0.8
        return {'success': success, 'actual_tool': actual_tool, 'param_score': param_score}

    def _fuzzy_match_params(self, actual: Dict, expected: Dict) -> float:
        if not expected:
            return 1.0
        matches = sum(1 for k, v in expected.items() if k in actual and (
            (isinstance(v, str) and isinstance(actual[k], str) and v.lower() in str(actual[k]).lower()) or
            actual[k] == v
        ))
        return matches / len(expected)

class RAGGroundednessValidator:
    async def validate(self, response_text: str, retrieved_chunks: List[str],
                      ground_truth_docs: List[str]) -> Dict[str, Any]:
        claims = [s.strip() for s in response_text.split('.') if len(s.strip()) > 10]
        supported = sum(1 for c in claims if self._check_claim_support(c, retrieved_chunks + ground_truth_docs))
        score = supported / len(claims) if claims else 1.0
        return {'score': score, 'unsupported_claims': [], 'total_claims': len(claims), 'supported_claims': supported}

    def _check_claim_support(self, claim: str, sources: List[str]) -> bool:
        cw = set(claim.lower().split())
        for s in sources:
            if len(cw & set(s.lower().split())) / max(len(cw), 1) > 0.5:
                return True
        return False
