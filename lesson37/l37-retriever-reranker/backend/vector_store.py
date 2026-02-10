import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import logging
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store with ChromaDB + BM25 for hybrid search"""
    
    def __init__(self):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.bm25 = None
        self.documents = []
        
    async def initialize(self):
        """Initialize vector store with sample documents"""
        try:
            self.client.delete_collection("retrieval_corpus")
        except:
            pass
        
        self.collection = self.client.create_collection("retrieval_corpus")
        
        # Sample documents for demonstration
        sample_docs = [
            {
                "chunk_id": "doc1_chunk1",
                "content": "Kubernetes HorizontalPodAutoscaler (HPA) automatically scales pods based on CPU or custom metrics. Configure with targetCPUUtilizationPercentage.",
                "metadata": {"source": "k8s_docs", "topic": "autoscaling"}
            },
            {
                "chunk_id": "doc1_chunk2",
                "content": "Cluster Autoscaler adjusts node count based on pod resource requests. Works with cloud providers like GKE, EKS, AKS.",
                "metadata": {"source": "k8s_docs", "topic": "autoscaling"}
            },
            {
                "chunk_id": "doc2_chunk1",
                "content": "Docker container orchestration manages lifecycle of containerized applications. Kubernetes provides declarative configuration.",
                "metadata": {"source": "docker_docs", "topic": "containers"}
            },
            {
                "chunk_id": "doc3_chunk1",
                "content": "RAG (Retrieval Augmented Generation) combines vector search with LLM generation for factual responses.",
                "metadata": {"source": "ai_docs", "topic": "rag"}
            },
            {
                "chunk_id": "doc3_chunk2",
                "content": "Cross-encoder rerankers score query-document pairs jointly, improving retrieval precision over bi-encoder embeddings.",
                "metadata": {"source": "ai_docs", "topic": "reranking"}
            },
            {
                "chunk_id": "doc4_chunk1",
                "content": "Production RAG systems use hybrid search: vector similarity for semantic matching, BM25 for keyword precision.",
                "metadata": {"source": "ai_docs", "topic": "rag"}
            },
            {
                "chunk_id": "doc5_chunk1",
                "content": "FastAPI supports async request handling with uvicorn ASGI server. Use asyncio.gather for parallel execution.",
                "metadata": {"source": "python_docs", "topic": "async"}
            },
        ]
        
        # Store documents
        self.documents = sample_docs
        
        # Generate embeddings
        contents = [doc["content"] for doc in sample_docs]
        embeddings = self.embedder.encode(contents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            ids=[doc["chunk_id"] for doc in sample_docs],
            embeddings=embeddings,
            documents=contents,
            metadatas=[doc["metadata"] for doc in sample_docs]
        )
        
        # Initialize BM25 for keyword search
        tokenized_docs = [self._tokenize(doc["content"]) for doc in sample_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Initialized vector store with {len(sample_docs)} documents")
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Vector similarity search"""
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.documents))
        )
        
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "chunk_id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "vector_score": float(1 - results['distances'][0][i])  # Convert distance to similarity
            })
        
        return chunks
    
    async def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 keyword search"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        chunks = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                chunks.append({
                    "chunk_id": doc["chunk_id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "bm25_score": float(scores[idx])
                })
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return re.findall(r'\w+', text.lower())
