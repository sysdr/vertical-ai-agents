import google.generativeai as genai
from typing import List, Dict, Any
import time
from backend.models.synthesis_models import (
    ReasoningStep, Chunk, ComplexityLevel
)

class CoTEngine:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def _assess_complexity(self, query: str, context: List[Chunk]) -> ComplexityLevel:
        """Determine query complexity for reasoning strategy selection"""
        # Simple heuristics
        query_length = len(query.split())
        context_count = len(context)
        
        if query_length < 10 and context_count <= 2:
            return ComplexityLevel.SIMPLE
        elif query_length < 25 and context_count <= 5:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.COMPLEX
    
    def plan_reasoning(self, query: str, context: List[Chunk]) -> List[Dict[str, str]]:
        """Decompose synthesis into explicit reasoning steps"""
        complexity = self._assess_complexity(query, context)
        
        if complexity == ComplexityLevel.SIMPLE:
            return [
                {"name": "direct_answer", "description": "Generate direct answer from context"}
            ]
        elif complexity == ComplexityLevel.MODERATE:
            return [
                {"name": "context_summary", "description": "Summarize key points from context"},
                {"name": "claim_formulation", "description": "Formulate response claims"},
                {"name": "citation_mapping", "description": "Map claims to sources"}
            ]
        else:  # COMPLEX
            return [
                {"name": "context_analysis", "description": "Analyze and categorize context"},
                {"name": "claim_decomposition", "description": "Break down query into sub-claims"},
                {"name": "evidence_mapping", "description": "Map evidence to each sub-claim"},
                {"name": "coherence_check", "description": "Verify logical consistency"},
                {"name": "final_assembly", "description": "Assemble coherent response"}
            ]
    
    def execute_step(self, step: Dict[str, str], query: str, context: List[Chunk], 
                    previous_outputs: List[Dict[str, Any]]) -> ReasoningStep:
        """Execute a single CoT reasoning step"""
        start_time = time.time()
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}: {c.source}]\n{c.content}" 
            for i, c in enumerate(context)
        ])
        
        # Build prompt based on step type
        if step["name"] == "direct_answer":
            prompt = f"""Given this context, provide a direct answer to the query.

Context:
{context_str}

Query: {query}

Provide a clear, concise answer grounded in the context."""

        elif step["name"] == "context_summary":
            prompt = f"""Summarize the key points from the following context relevant to the query.

Context:
{context_str}

Query: {query}

Provide a brief summary of the most relevant information."""

        elif step["name"] == "claim_formulation":
            prev_summary = previous_outputs[-1]["output"] if previous_outputs else ""
            prompt = f"""Based on the context summary, formulate clear claims that answer the query.

Summary: {prev_summary}

Query: {query}

List the main claims (2-5 claims) that directly address the query."""

        elif step["name"] == "citation_mapping":
            prev_claims = previous_outputs[-1]["output"] if previous_outputs else ""
            prompt = f"""For each claim, identify which source(s) support it.

Claims: {prev_claims}

Context:
{context_str}

For each claim, specify the supporting source(s)."""

        elif step["name"] == "context_analysis":
            prompt = f"""Analyze and categorize the provided context based on relevance to the query.

Context:
{context_str}

Query: {query}

Categorize the context into: Highly Relevant, Somewhat Relevant, Background Information."""

        elif step["name"] == "claim_decomposition":
            prompt = f"""Break down the query into sub-questions that need to be answered.

Query: {query}

List 2-4 sub-questions that together fully address the main query."""

        elif step["name"] == "evidence_mapping":
            prev_subclaims = previous_outputs[-1]["output"] if previous_outputs else ""
            prompt = f"""Map evidence from context to each sub-question.

Sub-questions: {prev_subclaims}

Context:
{context_str}

For each sub-question, identify supporting evidence."""

        elif step["name"] == "coherence_check":
            prev_evidence = previous_outputs[-1]["output"] if previous_outputs else ""
            prompt = f"""Check for logical consistency and contradictions in the evidence.

Evidence: {prev_evidence}

Identify any contradictions or gaps in reasoning."""

        else:  # final_assembly
            prev_evidence = previous_outputs[-2]["output"] if len(previous_outputs) >= 2 else ""
            prev_check = previous_outputs[-1]["output"] if previous_outputs else ""
            prompt = f"""Synthesize all evidence into a coherent, well-structured response.

Evidence: {prev_evidence}
Coherence Check: {prev_check}

Query: {query}

Provide a clear, comprehensive answer."""
        
        # Execute with Gemini
        response = self.model.generate_content(prompt)
        output = response.text
        
        # Calculate confidence (simplified - based on response length and coherence)
        confidence = min(0.95, 0.6 + len(output.split()) / 500)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ReasoningStep(
            step_name=step["name"],
            description=step["description"],
            input_data={"query": query, "context_count": len(context)},
            output_data={"output": output},
            confidence=confidence,
            latency_ms=latency_ms
        )
