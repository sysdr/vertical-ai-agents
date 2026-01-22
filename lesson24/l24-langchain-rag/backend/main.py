from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
import logging
import time
import json
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import os
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total RAG queries')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'RAG query duration')
RETRIEVAL_DURATION = Histogram('retrieval_duration_seconds', 'Retrieval duration')
GENERATION_DURATION = Histogram('generation_duration_seconds', 'Generation duration')

app = FastAPI(title="L24 LangChain RAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"  # hybrid, vector, keyword
    top_k: int = 5
    enable_reranking: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    chain_trace: List[Dict[str, Any]]

class DocumentInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}

USE_GEMINI = os.getenv("USE_GEMINI", "").lower() in {"1", "true", "yes"}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "INVALID_PLACEHOLDER")


class SimpleHashEmbeddings:
    """Lightweight, offline-friendly embedding function."""

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:32]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# Embeddings (prefer Gemini when explicitly enabled, otherwise offline-safe hash embeddings)
if USE_GEMINI and GEMINI_API_KEY and "INVALID_PLACEHOLDER" not in GEMINI_API_KEY:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
else:
    embeddings = SimpleHashEmbeddings()

# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="../data/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

try:
    collection = chroma_client.get_collection("l24_documents")
    logger.info("Loaded existing ChromaDB collection")
except:
    collection = chroma_client.create_collection(
        name="l24_documents",
        metadata={"description": "L24 LangChain RAG documents"}
    )
    logger.info("Created new ChromaDB collection")

# LangChain vectorstore
vectorstore = Chroma(
    client=chroma_client,
    collection_name="l24_documents",
    embedding_function=embeddings
)

# LLM (optional for offline/demo mode)
llm = None
if USE_GEMINI and GEMINI_API_KEY and "INVALID_PLACEHOLDER" not in GEMINI_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=1024
    )

# Custom Hybrid Retriever
class HybridChromaRetriever(BaseRetriever):
    """Custom retriever combining vector and keyword search"""
    
    vectorstore: Any
    search_kwargs: Dict[str, Any] = {"k": 10}
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search"""
        start = time.time()
        
        # Vector search
        vector_docs = self.vectorstore.similarity_search(
            query, 
            k=self.search_kwargs.get("k", 10)
        )
        
        # Keyword search via MMR
        try:
            keyword_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=self.search_kwargs.get("k", 10),
                fetch_k=20
            )
        except:
            keyword_docs = []
        
        # Fusion: combine and deduplicate
        seen = set()
        fused_docs = []
        
        # Interleave results
        max_len = max(len(vector_docs), len(keyword_docs))
        for i in range(max_len):
            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    fused_docs.append(doc)
            
            if i < len(keyword_docs):
                doc = keyword_docs[i]
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    fused_docs.append(doc)
        
        duration = time.time() - start
        RETRIEVAL_DURATION.observe(duration)
        
        return fused_docs[:self.search_kwargs.get("k", 10)]

# Initialize retrievers
hybrid_retriever = HybridChromaRetriever(
    vectorstore=vectorstore,
    search_kwargs={"k": 5}
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt template
template = """You are a helpful AI assistant. Answer the question based on the context provided.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context. If the context doesn't contain relevant information, say so.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Format documents helper
def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    return "\n\n".join([
        f"[Source {i+1}]\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])

def offline_answer(question: str, docs: List[Document]) -> str:
    """Fallback answer when external LLM access is disabled."""
    if not docs:
        return "Offline demo mode: no documents available yet."
    preview = "\n\n".join([
        f"[Source {i+1}] {doc.page_content[:200]}"
        for i, doc in enumerate(docs[:3])
    ])
    return (
        f"Offline demo answer for '{question}'.\n"
        f"Context used:\n{preview}"
    )

# Build chain using LCEL
def build_chain(retriever):
    """Build RAG chain using LangChain Expression Language"""
    if not llm:
        return None
    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Chain instances
hybrid_chain = build_chain(hybrid_retriever)
vector_chain = build_chain(vector_retriever)

# Callback handler for tracing
class TracingCallback(BaseCallbackHandler):
    """Callback to track chain execution"""
    
    def __init__(self):
        self.trace = []
        self.start_times = {}
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        step_id = kwargs.get("run_id")
        self.start_times[step_id] = time.time()
        self.trace.append({
            "step": "chain_start",
            "inputs": str(inputs)[:200],
            "timestamp": time.time()
        })
    
    def on_chain_end(self, outputs, **kwargs):
        step_id = kwargs.get("run_id")
        duration = time.time() - self.start_times.get(step_id, time.time())
        self.trace.append({
            "step": "chain_end",
            "duration": duration,
            "timestamp": time.time()
        })

# API endpoints
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "l24-langchain-rag"}

@app.post("/api/documents")
async def add_document(doc: DocumentInput):
    """Add document to vectorstore"""
    try:
        vectorstore.add_texts(
            texts=[doc.text],
            metadatas=[doc.metadata]
        )
        return {"status": "success", "message": "Document added"}
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG system using LangChain"""
    QUERY_COUNTER.inc()
    
    try:
        with QUERY_DURATION.time():
            # Select chain based on mode
            if request.mode == "hybrid":
                chain = hybrid_chain
                retriever = hybrid_retriever
            else:
                chain = vector_chain
                retriever = vector_retriever
            
            # Retrieve documents
            retrieval_start = time.time()
            docs = retriever.get_relevant_documents(request.query)
            retrieval_time = time.time() - retrieval_start
            
            # Generate answer
            generation_start = time.time()
            if chain:
                answer = chain.invoke(request.query)
            else:
                answer = offline_answer(request.query, docs)
            generation_time = time.time() - generation_start
            
            GENERATION_DURATION.observe(generation_time)
            
            # Format sources
            sources = [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata,
                    "relevance_score": 0.9 - (i * 0.1)
                }
                for i, doc in enumerate(docs)
            ]
            
            # Chain trace
            trace = [
                {"step": "retrieval", "duration": retrieval_time, "docs_found": len(docs)},
                {"step": "generation", "duration": generation_time, "tokens": len(answer.split())}
            ]
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                chain_trace=trace
            )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_documents": collection.count(),
        "retriever_type": "hybrid",
        "model": "gemini-2.0-flash-exp" if llm else "offline-demo",
        "embedding_model": "models/embedding-001" if isinstance(embeddings, GoogleGenerativeAIEmbeddings) else "hash-32"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
