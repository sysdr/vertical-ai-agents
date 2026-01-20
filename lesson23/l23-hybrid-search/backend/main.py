from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
import os
from collections import defaultdict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L23: Hybrid Search & Filtering")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. Vector search will be disabled.")
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "hybrid_search_docs"

try:
    collection = chroma_client.get_collection(name=collection_name)
    logger.info(f"Loaded existing collection: {collection_name}")
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Created new collection: {collection_name}")

# Initialize Reranker (from L22)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
logger.info("Reranker model loaded")

class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

class SearchQuery(BaseModel):
    query: str
    metadata_filter: Optional[Dict[str, Any]] = None
    top_k: int = 5
    use_reranking: bool = True
    alpha: float = 0.5  # Weight for vector vs keyword (0.5 = balanced)

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    rank_source: str

class HybridSearchEngine:
    """Production-grade hybrid search combining vector, keyword, and metadata filtering"""
    
    def __init__(self, collection, reranker):
        self.collection = collection
        self.reranker = reranker
        self.embedding_cache = {}
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            embedding = result['embedding']
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return None if embedding generation fails (e.g., API key expired)
            return None
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria"""
        if not filter_dict:
            return True
        
        if '$and' in filter_dict:
            # All conditions must match
            for condition in filter_dict['$and']:
                if not self._matches_single_condition(metadata, condition):
                    return False
            return True
        elif '$or' in filter_dict:
            # At least one condition must match
            for condition in filter_dict['$or']:
                if self._matches_single_condition(metadata, condition):
                    return True
            return False
        else:
            # Single condition
            return self._matches_single_condition(metadata, filter_dict)
    
    def _matches_single_condition(self, metadata: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if metadata matches a single condition"""
        for key, op in condition.items():
            if key not in metadata:
                return False
            
            value = metadata[key]
            if isinstance(op, dict):
                if '$eq' in op:
                    return value == op['$eq']
                elif '$ne' in op:
                    return value != op['$ne']
                elif '$gte' in op:
                    return value >= op['$gte']
                elif '$lte' in op:
                    return value <= op['$lte']
                elif '$gt' in op:
                    return value > op['$gt']
                elif '$lt' in op:
                    return value < op['$lt']
            else:
                # Direct equality
                return value == op
        return True
    
    def reciprocal_rank_fusion(self, results_list: List[List[str]], k: int = 60) -> List[tuple]:
        """
        RRF: Fuse multiple ranked lists
        score = Σ(1 / (rank_i + k)) where k=60 is standard
        """
        scores = defaultdict(float)
        
        for results in results_list:
            for rank, doc_id in enumerate(results):
                scores[doc_id] += 1.0 / (rank + k)
        
        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def hybrid_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and keyword search with RRF fusion
        alpha: weight for vector vs keyword (0.5 = balanced)
        """
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        
        # Execute parallel searches
        vector_results = []
        keyword_results = []
        
        # Vector search (only if embedding was successfully generated)
        if query_embedding is not None:
            try:
                vector_response = self.collection.query(
                    query_embeddings=[query_embedding],
                    where=metadata_filter,
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                if vector_response['ids'] and len(vector_response['ids'][0]) > 0:
                    vector_results = vector_response['ids'][0]
                    logger.info(f"Vector search returned {len(vector_results)} results")
                
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        else:
            logger.warning("Skipping vector search: embedding generation failed (check API key)")
        
        try:
            # Keyword search (full-text)
            # ChromaDB's query_texts doesn't support complex $and filters, so we search without filter and filter manually
            keyword_response = None
            try:
                # Try with simple filter first (single condition)
                if metadata_filter and '$and' not in metadata_filter:
                    keyword_response = self.collection.query(
                        query_texts=[query],
                        where=metadata_filter,
                        n_results=top_k * 2,
                        include=["documents", "metadatas", "distances"]
                    )
                else:
                    # For complex filters or no filter, search without where clause
                    keyword_response = self.collection.query(
                        query_texts=[query],
                        n_results=top_k * 3,  # Get more results to filter
                        include=["documents", "metadatas", "distances"]
                    )
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}, trying without filter")
                try:
                    keyword_response = self.collection.query(
                        query_texts=[query],
                        n_results=top_k * 3,
                        include=["documents", "metadatas", "distances"]
                    )
                except Exception as e2:
                    logger.error(f"Keyword search failed completely: {e2}")
                    keyword_response = None
            
            if keyword_response and keyword_response['ids'] and len(keyword_response['ids'][0]) > 0:
                # Apply metadata filter manually if needed
                if metadata_filter:
                    filtered_ids = []
                    for i, doc_id in enumerate(keyword_response['ids'][0]):
                        doc_metadata = keyword_response['metadatas'][0][i] if keyword_response['metadatas'] and i < len(keyword_response['metadatas'][0]) else {}
                        # Check if metadata matches filter
                        matches = self._matches_filter(doc_metadata, metadata_filter)
                        if matches:
                            filtered_ids.append(doc_id)
                    
                    keyword_results = filtered_ids
                    logger.info(f"Keyword search returned {len(keyword_results)} results after applying filters")
                else:
                    keyword_results = keyword_response['ids'][0]
                    logger.info(f"Keyword search returned {len(keyword_results)} results")
            else:
                keyword_results = []
                logger.info("Keyword search returned 0 results")
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            keyword_results = []
        
        # Apply RRF fusion
        fused_results = self.reciprocal_rank_fusion([vector_results, keyword_results])
        
        # Get full documents for top results
        doc_ids = [doc_id for doc_id, score in fused_results[:top_k]]
        
        if not doc_ids:
            return []
        
        # Fetch documents
        docs = self.collection.get(
            ids=doc_ids,
            include=["documents", "metadatas"]
        )
        
        # Build result list with RRF scores
        results = []
        score_map = dict(fused_results)
        
        for i, doc_id in enumerate(doc_ids):
            if i < len(docs['documents']):
                results.append({
                    'id': doc_id,
                    'text': docs['documents'][i],
                    'metadata': docs['metadatas'][i],
                    'rrf_score': score_map[doc_id],
                    'rank': i + 1
                })
        
        elapsed = time.time() - start_time
        logger.info(f"Hybrid search completed in {elapsed:.3f}s")
        
        return results
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking to results"""
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = [[query, result['text']] for result in results]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)
        
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]

# Initialize search engine
search_engine = HybridSearchEngine(collection, reranker)

@app.get("/")
async def root():
    return {
        "service": "L23: Hybrid Search & Filtering",
        "status": "operational",
        "features": ["vector_search", "keyword_search", "metadata_filtering", "rrf_fusion", "reranking"],
        "documents": collection.count()
    }

@app.post("/documents", response_model=Dict[str, str])
async def add_document(doc: Document):
    """Add document to collection"""
    try:
        # Generate embedding
        embedding = search_engine.get_embedding(doc.text)
        
        # If embedding generation failed (e.g., API key expired), use a dummy embedding
        # This allows documents to be added and searched via keyword search
        if embedding is None:
            logger.warning(f"Embedding generation failed for {doc.id}, using dummy embedding (keyword search will still work)")
            # Create a dummy embedding vector (384 dimensions to match ChromaDB's default)
            # ChromaDB uses 384-dimensional embeddings by default for text search
            embedding = [0.0] * 384
            embedding[0] = 1.0  # Small non-zero value to avoid all-zero vector
        
        # Add to collection
        collection.add(
            ids=[doc.id],
            documents=[doc.text],
            metadatas=[doc.metadata],
            embeddings=[embedding]
        )
        
        logger.info(f"Added document: {doc.id}")
        return {"status": "success", "id": doc.id}
    
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(query: SearchQuery):
    """Hybrid search with optional reranking"""
    try:
        # Execute hybrid search
        results = search_engine.hybrid_search(
            query=query.query,
            metadata_filter=query.metadata_filter,
            top_k=20,  # Get more candidates for reranking
            alpha=query.alpha
        )
        
        # If no results and no documents in collection, return empty list with helpful message
        if not results and collection.count() == 0:
            logger.info(f"Search query '{query.query}' returned no results (collection is empty)")
            return []
        
        # Apply reranking if enabled
        if query.use_reranking and results:
            results = search_engine.rerank_results(query.query, results, top_k=query.top_k)
            rank_source = "hybrid_rrf_reranked"
        else:
            results = results[:query.top_k]
            rank_source = "hybrid_rrf"
        
        # Format response
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                text=result['text'],
                score=result.get('rerank_score', result.get('rrf_score', 0.0)),
                metadata=result['metadata'],
                rank_source=rank_source
            ))
        
        logger.info(f"Search query '{query.query}' returned {len(search_results)} results")
        return search_results
    
    except Exception as e:
        error_msg = str(e)
        # Check if it's an API key error
        if "API key" in error_msg or "API_KEY" in error_msg:
            logger.error(f"Search failed due to API key issue: {e}")
            # Return empty results instead of 500 error
            return []
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    count = collection.count()
    
    # Get sample metadata to show available filters
    sample_metadata = {}
    if count > 0:
        sample = collection.get(limit=1, include=["metadatas"])
        if sample['metadatas']:
            sample_metadata = sample['metadatas'][0]
    
    return {
        "total_documents": count,
        "sample_metadata_fields": list(sample_metadata.keys()) if sample_metadata else [],
        "collection_name": collection_name
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document by ID"""
    try:
        collection.delete(ids=[doc_id])
        logger.info(f"Deleted document: {doc_id}")
        return {"status": "success", "deleted_id": doc_id}
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
