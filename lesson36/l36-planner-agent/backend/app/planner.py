"""PlannerAgent - Query Decomposition Engine"""
import json
import logging
from typing import Optional
import google.generativeai as genai
from .models import QueryPlan, SubQuery, PlanRequest
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """You are a query planning expert for enterprise RAG systems. Decompose complex queries into optimal sub-queries for retrieval.

Query: {query}

Analyze this query and return JSON with this EXACT structure (no markdown, no explanations):
{{
  "requires_decomposition": boolean,
  "sub_queries": [
    {{"text": "sub-query 1", "rationale": "why this helps", "requires_previous": false}},
    {{"text": "sub-query 2", "rationale": "why this helps", "requires_previous": false}}
  ],
  "strategy": "parallel or sequential",
  "estimated_complexity": "low or medium or high"
}}

RULES:
- If query is simple (single concept, direct question), set requires_decomposition=false and return 1 sub-query with original text
- If query is complex (multiple concepts, comparisons, analysis), set requires_decomposition=true
- Maximum {max_sub_queries} sub-queries
- Each sub-query must be independently searchable
- Use "parallel" strategy unless sub-queries have dependencies (then "sequential")
- Set requires_previous=true only if a sub-query needs results from the previous one
- Complexity: low (1 concept), medium (2-3 concepts), high (4+ concepts or deep analysis)

Return ONLY the JSON, no other text."""

class PlannerAgent:
    """Query decomposition agent using Gemini AI"""
    
    def __init__(self, api_key: str, timeout: float = 5.0):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.timeout = timeout
        logger.info("PlannerAgent initialized with Gemini 2.0 Flash")
    
    async def decompose_query(self, request: PlanRequest) -> QueryPlan:
        """Decompose query into searchable sub-queries"""
        try:
            # Create decomposition prompt
            prompt = DECOMPOSITION_PROMPT.format(
                query=request.query,
                max_sub_queries=request.max_sub_queries
            )
            
            # Call Gemini with timeout
            logger.info(f"Decomposing query: {request.query[:100]}...")
            
            # Run with timeout
            response = await asyncio.wait_for(
                self._call_gemini(prompt),
                timeout=self.timeout
            )
            
            # Parse JSON response
            plan_data = self._parse_llm_response(response)
            
            # Validate and construct QueryPlan
            plan = QueryPlan(
                original_query=request.query,
                requires_decomposition=plan_data["requires_decomposition"],
                sub_queries=[SubQuery(**sq) for sq in plan_data["sub_queries"]],
                strategy=plan_data["strategy"],
                estimated_complexity=plan_data["estimated_complexity"]
            )
            
            logger.info(f"Decomposition complete: {len(plan.sub_queries)} sub-queries, strategy={plan.strategy}")
            return plan
            
        except asyncio.TimeoutError:
            logger.warning(f"Decomposition timeout for query: {request.query[:100]}")
            return self._fallback_plan(request.query)
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            return self._fallback_plan(request.query)
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API (async wrapper)"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(prompt)
        )
        return response.text
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse and validate LLM JSON response"""
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Extract JSON from markdown
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```"):
                    if in_json:
                        break
                    in_json = True
                    continue
                if in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)
        
        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM: {response[:200]}")
            raise ValueError(f"LLM returned invalid JSON: {str(e)}")
        
        # Validate required fields
        required = ["requires_decomposition", "sub_queries", "strategy", "estimated_complexity"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return data
    
    def _fallback_plan(self, query: str) -> QueryPlan:
        """Create fallback plan when decomposition fails"""
        logger.info("Using fallback plan (no decomposition)")
        return QueryPlan(
            original_query=query,
            requires_decomposition=False,
            sub_queries=[
                SubQuery(
                    text=query,
                    rationale="Fallback to original query due to decomposition failure",
                    requires_previous=False
                )
            ],
            strategy="parallel",
            estimated_complexity="medium"
        )
