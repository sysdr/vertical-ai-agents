import google.generativeai as genai
from models import QueryPlan, FailureSignal, FailureType
from typing import Optional

class EnhancedPlannerAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def plan(self, query: str) -> QueryPlan:
        """Create initial query plan"""
        return QueryPlan(
            original_query=query,
            reformulated_query=None,
            strategy="standard"
        )
    
    async def reformulate(self, original_query: str, failure_signal: FailureSignal) -> QueryPlan:
        """Reformulate query based on failure signal"""
        reformulated = original_query
        strategy = "retry"
        
        if failure_signal.type == FailureType.COMPLIANCE_VIOLATION:
            # Add filtering keywords
            reformulated = f"{original_query} (exclude: sensitive, restricted, confidential)"
            strategy = "compliance_filter"
        
        elif failure_signal.type == FailureType.FACTUAL_INCONSISTENCY:
            # Request explicit sources
            reformulated = f"{original_query} - provide specific sources and citations"
            strategy = "source_attribution"
        
        elif failure_signal.type == FailureType.INSUFFICIENT_CONTEXT:
            # Decompose into sub-queries
            prompt = f"Break down this query into 2-3 specific sub-queries: {original_query}"
            try:
                response = self.model.generate_content(prompt)
                sub_queries = [q.strip() for q in response.text.split('\n') if q.strip()][:3]
                return QueryPlan(
                    original_query=original_query,
                    reformulated_query=None,
                    strategy="sub_query_decomposition",
                    sub_queries=sub_queries
                )
            except:
                pass
        
        return QueryPlan(
            original_query=original_query,
            reformulated_query=reformulated,
            strategy=strategy
        )
