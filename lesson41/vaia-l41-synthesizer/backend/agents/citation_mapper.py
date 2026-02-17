import google.generativeai as genai
from typing import List, Dict
import re
from backend.models.synthesis_models import Chunk, Citation

class CitationMapper:
    def __init__(self, api_key: str, model_name: str, granularity: str = "medium"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.granularity = granularity
        
    def map_claims_to_sources(self, response_text: str, chunks: List[Chunk]) -> List[Citation]:
        """Map response claims to source chunks"""
        # Build context for mapping
        context_str = "\n\n".join([
            f"[Chunk {i+1} ID: {c.id}]\nSource: {c.source}\n{c.content}" 
            for i, c in enumerate(chunks)
        ])
        
        prompt = f"""Analyze the response and identify which chunks support each factual claim.

Response:
{response_text}

Context Chunks:
{context_str}

For each factual claim in the response:
1. Extract the claim text (up to 100 chars)
2. Identify the chunk ID that supports it
3. Extract a brief excerpt from the chunk

Format each citation as:
CITATION: [claim text] | [chunk_id] | [excerpt from chunk]

List all citations:"""

        result = self.model.generate_content(prompt)
        citation_text = result.text
        
        # Parse citations
        citations = []
        pattern = r'CITATION:\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(.+?)(?=\nCITATION:|\Z)'
        matches = re.findall(pattern, citation_text, re.DOTALL)
        
        for claim, chunk_id, excerpt in matches:
            # Find the actual chunk
            chunk = next((c for c in chunks if c.id == chunk_id.strip()), None)
            if chunk:
                citations.append(Citation(
                    chunk_id=chunk_id.strip(),
                    source=chunk.source,
                    section=chunk.section,
                    text_excerpt=excerpt.strip()[:200],
                    confidence=chunk.quality_score if chunk.quality_score > 0 else 0.8
                ))
        
        return citations
    
    def inject_citations(self, response: str, citations: List[Citation]) -> str:
        """Add inline citations to response text"""
        # Group citations by source
        citation_refs = {}
        for i, cit in enumerate(citations, 1):
            ref = f"[{i}]"
            citation_refs[cit.text_excerpt[:50]] = ref
        
        # Append citation references at end
        cited_response = response + "\n\n---\nSources:\n"
        for i, cit in enumerate(citations, 1):
            cited_response += f"[{i}] {cit.source}"
            if cit.section:
                cited_response += f", §{cit.section}"
            cited_response += "\n"
        
        return cited_response
