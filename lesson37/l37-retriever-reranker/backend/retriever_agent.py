import asyncio
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RetrieverAgent:
    """
    RetrieverAgent orchestrates the complete retrieval pipeline:
    - Executes parallel vector searches for each sub-query
    - Merges and deduplicates results
    - Applies reranking for semantic relevance boosting
    - Returns top-K globally ranked chunks
    """
    
    def __init__(self, vector_store, reranker):
        self.vector_store = vector_store
        self.reranker = reranker
        self.retrieval_count = 0
        
    async def retrieve(
        self,
        original_query: str,
        sub_queries: List[str],
        top_k: int = 10,
        use_reranking: bool = True,
        use_hybrid: bool = False
    ) -> List[Dict]:
        """
        Execute multi-stage retrieval pipeline
        
        Args:
            original_query: Original user query (used for final reranking)
            sub_queries: Decomposed sub-queries from PlannerAgent
            top_k: Number of final results to return
            use_reranking: Whether to apply reranking
            use_hybrid: Whether to use hybrid (vector + keyword) search
        
        Returns:
            List of ranked chunks with scores
        """
        self.retrieval_count += 1
        logger.info(f"Retrieval #{self.retrieval_count}: {len(sub_queries)} sub-queries")
        
        # Stage 1: Parallel vector searches
        search_tasks = []
        for sq in sub_queries:
            if use_hybrid:
                task = self._hybrid_search(sq, top_k=100)
            else:
                task = self.vector_store.search(sq, top_k=100)
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Handle search failures gracefully
        valid_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for sub-query {i}: {result}")
            else:
                valid_results.extend(result)
        
        if not valid_results:
            logger.warning("All searches failed, returning empty results")
            return []
        
        # Stage 2: Merge and deduplicate
        merged_chunks = self._merge_deduplicate(valid_results)
        logger.info(f"Merged to {len(merged_chunks)} unique chunks")
        
        # Stage 3: Reranking (if enabled)
        if use_reranking and merged_chunks:
            try:
                reranked_chunks = await self.reranker.rerank(
                    query=original_query,
                    chunks=merged_chunks,
                    top_k=top_k
                )
                logger.info(f"Reranked to top-{len(reranked_chunks)}")
                return reranked_chunks
            except Exception as e:
                logger.error(f"Reranking failed: {e}, falling back to vector scores")
                # Fallback: return vector-scored results
                return sorted(
                    merged_chunks,
                    key=lambda x: x.get("vector_score", 0),
                    reverse=True
                )[:top_k]
        else:
            # No reranking: return top-K by vector score
            return sorted(
                merged_chunks,
                key=lambda x: x.get("vector_score", 0),
                reverse=True
            )[:top_k]
    
    async def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Execute hybrid search: vector + keyword (BM25)
        Merge results using Reciprocal Rank Fusion (RRF)
        """
        # Parallel execution of vector and keyword search
        vector_task = self.vector_store.search(query, top_k=top_k)
        keyword_task = self.vector_store.keyword_search(query, top_k=top_k)
        
        vector_results, keyword_results = await asyncio.gather(
            vector_task,
            keyword_task,
            return_exceptions=True
        )
        
        if isinstance(vector_results, Exception):
            vector_results = []
        if isinstance(keyword_results, Exception):
            keyword_results = []
        
        # Apply RRF (Reciprocal Rank Fusion)
        return self._reciprocal_rank_fusion(vector_results, keyword_results, top_k)
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Merge results using RRF: score(doc) = Σ(1 / (k + rank_i))
        
        Args:
            k: RRF constant (typically 60)
        """
        scores = {}
        
        # Score vector results
        for rank, chunk in enumerate(vector_results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, {"chunk": chunk, "score": 0})
            scores[chunk_id]["score"] += 1 / (k + rank)
        
        # Score keyword results
        for rank, chunk in enumerate(keyword_results, start=1):
            chunk_id = chunk["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {"chunk": chunk, "score": 0}
            scores[chunk_id]["score"] += 1 / (k + rank)
        
        # Sort by RRF score and return top-K
        sorted_chunks = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        # Add RRF score to chunk metadata
        result_chunks = []
        for item in sorted_chunks:
            chunk = item["chunk"].copy()
            chunk["rrf_score"] = item["score"]
            result_chunks.append(chunk)
        
        return result_chunks
    
    def _merge_deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """
        Merge chunks from multiple searches and deduplicate
        Uses content hash to identify duplicates
        """
        seen = {}
        for chunk in chunks:
            content_hash = hash(chunk["content"])
            if content_hash not in seen:
                seen[content_hash] = chunk
            else:
                # Keep chunk with higher score
                if chunk.get("vector_score", 0) > seen[content_hash].get("vector_score", 0):
                    seen[content_hash] = chunk
        
        return list(seen.values())
