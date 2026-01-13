from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def search(
        self,
        query: str,
        collection_name: str = "default",
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional metadata filtering"""
        # Embed query
        query_embedding = await self.embedding_service.embed_query(query)
        
        # Search vector store
        raw_results = self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )
        
        # Format results
        results = []
        for i in range(len(raw_results['ids'][0])):
            results.append({
                "id": raw_results['ids'][0][i],
                "document": raw_results['documents'][0][i],
                "metadata": raw_results['metadatas'][0][i],
                "distance": raw_results['distances'][0][i],
                "similarity": 1 - raw_results['distances'][0][i]  # Cosine similarity
            })
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}")
        return results
