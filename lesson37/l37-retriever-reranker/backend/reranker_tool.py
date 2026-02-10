import os
import google.generativeai as genai
from typing import List, Dict
import json
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

class GeminiReranker:
    """
    Cross-encoder reranker using Gemini AI
    Scores query-document pairs for semantic relevance
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.rerank_count = 0
        self.batch_size = 50  # Maximum chunks per API call
        
    async def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 10,
        timeout_ms: int = 500
    ) -> List[Dict]:
        """
        Rerank chunks using cross-encoder scoring
        
        Args:
            query: Original user query
            chunks: List of retrieved chunks
            top_k: Number of top results to return
            timeout_ms: Maximum time for reranking (fallback to vector scores if exceeded)
        
        Returns:
            Top-K reranked chunks with rerank_score field
        """
        self.rerank_count += 1
        start_time = datetime.now()
        
        try:
            # Split into batches if needed
            if len(chunks) <= self.batch_size:
                scored_chunks = await self._score_batch(query, chunks)
            else:
                # Process in parallel batches
                batches = [
                    chunks[i:i + self.batch_size]
                    for i in range(0, len(chunks), self.batch_size)
                ]
                
                batch_tasks = [
                    self._score_batch(query, batch)
                    for batch in batches
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Merge batch results
                scored_chunks = []
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch scoring failed: {result}")
                    else:
                        scored_chunks.extend(result)
            
            # Check timeout
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > timeout_ms:
                logger.warning(f"Reranking timeout ({elapsed_ms:.0f}ms > {timeout_ms}ms), using partial results")
            
            # Sort by rerank score and return top-K
            sorted_chunks = sorted(
                scored_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )[:top_k]
            
            logger.info(f"Reranked {len(chunks)} chunks to top-{len(sorted_chunks)} in {elapsed_ms:.0f}ms")
            return sorted_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return chunks sorted by vector score
            return sorted(chunks, key=lambda x: x.get("vector_score", 0), reverse=True)[:top_k]
    
    async def _score_batch(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Score a single batch of chunks"""
        
        # Format documents for Gemini
        docs_text = "\n\n".join([
            f"[DOC_{i}]\n{chunk['content'][:500]}"  # Limit to 500 chars per doc
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""You are a semantic relevance scorer. Score each document's relevance to the query on a scale of 0-100.
- 0: Completely irrelevant
- 50: Somewhat related but not directly answering
- 100: Highly relevant and directly answers the query

Query: {query}

Documents:
{docs_text}

Return ONLY a JSON array of scores in order: [85, 72, 91, ...]
No explanation, just the array."""

        try:
            # Call Gemini API
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=1000
                )
            )
            
            # Parse scores
            scores_text = response.text.strip()
            if scores_text.startswith("```"):
                scores_text = scores_text.split("```")[1].replace("json", "").strip()
            
            scores = json.loads(scores_text)
            
            if len(scores) != len(chunks):
                logger.warning(f"Score count mismatch: {len(scores)} != {len(chunks)}")
                # Pad with zeros if needed
                scores.extend([0] * (len(chunks) - len(scores)))
            
            # Attach scores to chunks
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                scored_chunk = chunk.copy()
                scored_chunk["rerank_score"] = float(score)
                scored_chunks.append(scored_chunk)
            
            return scored_chunks
            
        except Exception as e:
            logger.error(f"Batch scoring error: {e}")
            # Return chunks with zero rerank scores
            return [
                {**chunk, "rerank_score": 0}
                for chunk in chunks
            ]
