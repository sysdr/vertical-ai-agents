import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import asyncio
import os
import threading
from typing import List, Dict
import logging
from llm_client import generate_with_retry_async

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
logger = logging.getLogger(__name__)

class RAGEngine:
    """Simplified RAG engine integrating L19-L27 components"""
    
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PATH", "./data/chromadb")
        )
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self._get_or_create_collection()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        self.model = genai.GenerativeModel(model_name)
        self._ensure_sample_data()
        self._prewarm_embeddings()
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(
                name="rag_knowledge",
                embedding_function=self.embedding_fn
            )
        except:
            return self.chroma_client.create_collection(
                name="rag_knowledge",
                embedding_function=self.embedding_fn
            )
    
    def _ensure_sample_data(self):
        """Add sample documents if collection empty"""
        if self.collection.count() == 0:
            docs = [
                "Python is a high-level programming language known for readability.",
                "FastAPI is a modern web framework for building APIs with Python.",
                "Machine learning models require training data and evaluation metrics.",
                "Vector databases store embeddings for similarity search.",
                "RAG systems combine retrieval with generation for grounded responses."
            ]
            
            self.collection.add(
                documents=docs,
                ids=[f"doc_{i}" for i in range(len(docs))]
            )
            logger.info(f"Added {len(docs)} sample documents")
    
    def _prewarm_embeddings(self):
        """Load embedding model in background so first query is fast."""
        def _run():
            try:
                self.collection.query(query_texts=["prewarm"], n_results=1)
                logger.info("RAG embedding model pre-warmed")
            except Exception as e:
                logger.warning(f"RAG prewarm failed (will load on first query): {e}")
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    
    async def query(
        self, 
        query: str, 
        top_k: int = 5,
        rerank_top_k: int = 3
    ) -> Dict:
        """Execute RAG query with retrieval and generation"""
        
        # Retrieve relevant documents
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results["documents"][0]
        ids = results["ids"][0]
        
        # Simple reranking (take top_k)
        reranked_docs = documents[:rerank_top_k]
        reranked_ids = ids[:rerank_top_k]
        
        # Build context
        context = "\n\n".join([
            f"[{i+1}] {doc}" 
            for i, doc in enumerate(reranked_docs)
        ])
        
        # Generate answer (with fallback when API quota exceeded or timeout)
        rag_timeout = int(os.getenv("RAG_LLM_TIMEOUT", "25"))
        try:
            prompt = f"""Answer this question using ONLY the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer with source citations."""
            answer = await asyncio.wait_for(
                generate_with_retry_async(prompt),
                timeout=rag_timeout
            )
        except asyncio.TimeoutError:
            logger.info("RAG LLM timeout, using retrieval-only fallback")
            answer = f"Based on the retrieved documents:\n\n{context}\n\n(Note: AI synthesis timed out. Above is the relevant retrieved context.)"
        except Exception as e:
            err_str = str(e).lower()
            is_api_err = ("429" in str(e) or "quota" in err_str or "credentials" in err_str or
                          ("api" in err_str and "key" in err_str) or "expired" in err_str)
            if is_api_err:
                logger.info(f"RAG LLM unavailable, using retrieval-only fallback")
                answer = f"Based on the retrieved documents:\n\n{context}\n\n(Note: Full AI synthesis unavailable due to API limits. Above is the relevant retrieved context.)"
            else:
                raise
        return {
            "answer": answer,
            "context": context,
            "sources": reranked_ids,
            "num_sources": len(reranked_ids)
        }
