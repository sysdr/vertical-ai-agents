from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import re
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

app = FastAPI(title="L25 Advanced Prompting RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration: prefer env so users can set GEMINI_API_KEY or GOOGLE_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
) if GEMINI_API_KEY else None

# Use a supported model (gemini-1.5-flash returns 404; gemini-2.0-flash / gemini-2.5-flash are current)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
) if GEMINI_API_KEY else None

# Prompt Templates Library
PROMPT_TEMPLATES = {
    "strict": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a precise information assistant operating under STRICT GROUNDING rules.

RETRIEVED CONTEXT:
{context}

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in RETRIEVED CONTEXT above
2. Include citations for every claim: [Source 1], [Source 2], etc.
3. If RETRIEVED CONTEXT does not contain information needed to answer the question, you MUST respond EXACTLY: "The provided documents do not contain information about [specific topic]."
4. NEVER EVER use your training knowledge to supplement or enhance the RETRIEVED CONTEXT
5. Do not make inferences beyond what is explicitly stated
6. If context is irrelevant to the question, refuse to answer

Question: {question}

Answer with inline citations [Source N]:"""
    ),
    
    "moderate": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant that prioritizes retrieved context.

RETRIEVED CONTEXT:
{context}

GUIDELINES:
1. Base your answer primarily on RETRIEVED CONTEXT
2. Include citations: [Source 1], [Source 2] for context-based claims
3. You may add brief general knowledge for context, but clearly distinguish it
4. If RETRIEVED CONTEXT is insufficient, state what's missing before adding general info
5. Mark general knowledge with phrases like "Generally speaking..." or "In general..."

Question: {question}

Answer:"""
    ),
    
    "creative": PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a creative assistant that uses context as inspiration.

RETRIEVED CONTEXT (use as foundation):
{context}

APPROACH:
1. Start with information from RETRIEVED CONTEXT
2. Expand with relevant knowledge and insights
3. Cite sources when directly using context: [Source 1]
4. Clearly distinguish between context-based and knowledge-based information

Question: {question}

Answer:"""
    )
}

# Grounding Validator
class GroundingValidator:
    """Validates context relevance before generation"""
    
    @staticmethod
    def is_relevant(question: str, context: str, threshold: float = 0.3) -> Dict:
        """Check if context is relevant to question"""
        # Simple keyword overlap heuristic (normalize: strip punctuation so "rag?" matches "rag")
        norm = lambda s: set(re.sub(r"[^\w\s]", " ", s.lower()).split())
        question_words = norm(question)
        context_words = norm(context)
        overlap = len(question_words & context_words)
        total = len(question_words)
        
        relevance_score = overlap / total if total > 0 else 0
        
        return {
            "is_relevant": relevance_score >= threshold,
            "relevance_score": relevance_score,
            "reason": f"Keyword overlap: {relevance_score:.2%}"
        }

# Citation Extractor
class CitationExtractor:
    """Validates and extracts citations from generated answers"""
    
    @staticmethod
    def extract_citations(answer: str) -> List[int]:
        """Extract citation numbers from answer"""
        pattern = r'\[Source (\d+)\]'
        citations = re.findall(pattern, answer)
        return [int(c) for c in citations]
    
    @staticmethod
    def validate_citations(answer: str, num_sources: int) -> Dict:
        """Validate that all citations reference existing sources"""
        citations = CitationExtractor.extract_citations(answer)
        
        invalid = [c for c in citations if c < 1 or c > num_sources]
        
        return {
            "valid": len(invalid) == 0,
            "citations_found": citations,
            "invalid_citations": invalid,
            "num_sources": num_sources
        }

# Adversarial Tester
class AdversarialTester:
    """Tests grounding strength with misleading contexts"""
    
    @staticmethod
    def generate_misleading_context(question: str) -> str:
        """Generate intentionally irrelevant context"""
        misleading_contexts = {
            "python": "JavaScript promises and async/await were introduced in ES6...",
            "revenue": "The company's 2022 Q1 revenue showed strong growth...",
            "api": "REST APIs use HTTP methods like GET, POST, PUT, DELETE...",
            "database": "NoSQL databases like MongoDB use document-based storage...",
            "default": "The history of ancient Roman architecture spans centuries..."
        }
        
        q_lower = question.lower()
        for keyword, context in misleading_contexts.items():
            if keyword in q_lower:
                return context
        
        return misleading_contexts["default"]

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    domain: str = "strict"  # strict, moderate, creative
    test_mode: bool = False  # If True, use misleading context

class Citation(BaseModel):
    source_id: int
    text: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    grounding_state: str  # grounded, partial, rejected
    citations: List[Citation]
    grounding_confidence: float
    validation_results: Dict
    metadata: Dict

class PromptTestRequest(BaseModel):
    question: str
    context: str
    template_name: str

# Global state
vector_store = None
retrieval_chains = {}

def initialize_vector_store():
    """Initialize ChromaDB vector store with sample documents"""
    global vector_store, retrieval_chains
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Sample knowledge base documents
    documents = [
        "Gemini AI API supports vision, text, and multimodal capabilities with a 1M token context window.",
        "RAG systems combine retrieval with generation to ground LLM responses in factual documents.",
        "Python 3.12 introduced improved error messages and performance optimizations.",
        "FastAPI is a modern web framework for building APIs with Python 3.6+ based on type hints.",
        "Vector databases enable semantic search by storing embeddings of documents.",
        "ChromaDB is an open-source embedding database for AI applications.",
        "LangChain provides abstractions for building LLM applications including RAG pipelines.",
        "System prompts guide LLM behavior and can enforce grounding constraints.",
        "Citation tracking maps generated claims back to source documents for verification.",
        "Adversarial testing with irrelevant context measures grounding strength of RAG systems."
    ]
    metadatas = [{"source": f"doc_{i+1}", "id": i+1} for i in range(len(documents))]

    emb = embeddings
    if emb is None:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_store = Chroma.from_texts(
            texts=documents,
            embedding=emb,
            metadatas=metadatas,
            collection_name="l25_knowledge"
        )
    except Exception:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_texts(
            texts=documents,
            embedding=emb,
            metadatas=metadatas,
            collection_name="l25_knowledge"
        )

    # Initialize retrieval chains only when LLM is available
    if llm is not None:
        for template_name, template in PROMPT_TEMPLATES.items():
            retrieval_chains[template_name] = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": template},
                return_source_documents=True
            )

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_vector_store()
    print("✓ Vector store initialized")
    print("✓ Retrieval chains configured")
    print("✓ System ready")

@app.get("/")
async def root():
    return {
        "service": "L25 Advanced Prompting RAG",
        "status": "operational",
        "templates": list(PROMPT_TEMPLATES.keys()),
        "timestamp": datetime.now().isoformat()
    }

def _demo_mode_response(msg: str, **kwargs) -> QueryResponse:
    """Return 200 with a QueryResponse so the frontend never sees 503."""
    return QueryResponse(
        answer=msg,
        grounding_state=kwargs.get("grounding_state", "rejected"),
        citations=kwargs.get("citations", []),
        grounding_confidence=kwargs.get("grounding_confidence", 0.0),
        validation_results=kwargs.get("validation_results", {"demo_mode": True}),
        metadata=kwargs.get("metadata", {"demo_mode": True})
    )

def _retrieval_only_response(question: str, domain: str, quota_exceeded: bool = False) -> Optional[QueryResponse]:
    """When LLM fails (e.g. 429 quota), return 200 with retrieval-only answer so dashboard stays useful."""
    if not vector_store:
        return None
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        source_docs = retriever.invoke(question)
        grounding_check = GroundingValidator.is_relevant(question, "\n\n".join(doc.page_content for doc in source_docs))
        parts = [f"[Source {i}] {doc.page_content}" for i, doc in enumerate(source_docs, 1)]
        body = "Based on the retrieved documents:\n\n" + "\n\n".join(parts)
        note = "Quota or rate limit reached; showing retrieval-only answer. Retry in a minute or see https://ai.google.dev/gemini-api/docs/rate-limits\n\n" if quota_exceeded else ""
        answer = note + body
        citations = [Citation(source_id=i, text=doc.page_content[:100] + "...", relevance_score=0.5) for i, doc in enumerate(source_docs, 1)]
        confidence = grounding_check["relevance_score"] if grounding_check["is_relevant"] else 0.0
        grounding_state = "grounded" if grounding_check["is_relevant"] else "partial"
        return QueryResponse(
            answer=answer,
            grounding_state=grounding_state,
            citations=citations,
            grounding_confidence=confidence,
            validation_results={"grounding_check": grounding_check, "quota_fallback": quota_exceeded},
            metadata={"domain": domain, "num_sources": len(source_docs), "quota_exceeded": quota_exceeded}
        )
    except Exception:
        return None

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute RAG query with advanced prompting. Never returns 503 — uses demo mode when no API key."""
    # Demo mode: no API key — use retrieval only; always return 200
    if not retrieval_chains:
        if not vector_store:
            return _demo_mode_response(
                "Service is still initializing. Please refresh the page or try again in a few seconds."
            )
        if request.test_mode:
            return _demo_mode_response(
                "The provided documents do not contain information about that topic. (Demo mode: set GEMINI_API_KEY for full adversarial tests.)",
                metadata={"domain": request.domain, "demo_mode": True}
            )
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            source_docs = retriever.invoke(request.question)
            context_combined = "\n\n".join(doc.page_content for doc in source_docs)
            grounding_check = GroundingValidator.is_relevant(request.question, context_combined)
            parts = [f"[Source {i}] {doc.page_content}" for i, doc in enumerate(source_docs, 1)]
            answer = "Based on the retrieved documents:\n\n" + "\n\n".join(parts)
            citations = [
                Citation(source_id=i, text=doc.page_content[:100] + "...", relevance_score=0.5)
                for i, doc in enumerate(source_docs, 1)
            ]
            confidence = grounding_check["relevance_score"] if grounding_check["is_relevant"] else 0.0
            grounding_state = "grounded" if grounding_check["is_relevant"] else "partial"
            return QueryResponse(
                answer=answer,
                grounding_state=grounding_state,
                citations=citations,
                grounding_confidence=confidence,
                validation_results={"grounding_check": grounding_check, "demo_mode": True},
                metadata={"domain": request.domain, "num_sources": len(source_docs), "demo_mode": True}
            )
        except Exception as e:
            return _demo_mode_response(
                f"Retrieval temporarily unavailable: {e!s}. Please try again or set GEMINI_API_KEY for full RAG."
            )
    # Get appropriate retrieval chain (when API key is set)
    if request.domain not in retrieval_chains:
        raise HTTPException(400, f"Invalid domain. Choose from: {list(PROMPT_TEMPLATES.keys())}")
    
    chain = retrieval_chains[request.domain]
    
    try:
        # Test mode: use misleading context
        if request.test_mode:
            misleading_context = AdversarialTester.generate_misleading_context(request.question)
            prompt = PROMPT_TEMPLATES[request.domain].format(
                context=misleading_context,
                question=request.question
            )
            result_text = llm.invoke(prompt).content
            return QueryResponse(
                answer=result_text,
                grounding_state="rejected" if "do not contain" in result_text.lower() else "hallucinated",
                citations=[],
                grounding_confidence=0.0,
                validation_results={"test_mode": True, "passed": "do not contain" in result_text.lower()},
                metadata={"domain": request.domain, "adversarial_test": True}
            )
        # Normal mode: use actual retrieval
        result = chain.invoke({"query": request.question})
    except Exception as e:
        err_str = str(e).lower()
        is_quota = "429" in str(e) or "quota" in err_str or "rate limit" in err_str or "rate-limits" in err_str
        if is_quota and vector_store:
            fallback = _retrieval_only_response(request.question, request.domain, quota_exceeded=True)
            if fallback is not None:
                return fallback
        return _demo_mode_response(
            f"LLM or API error: {e!s}. Check GEMINI_API_KEY in .env and run ./stop.sh then ./start.sh."
        )
    
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    # Validate grounding
    context_combined = "\n\n".join([doc.page_content for doc in source_docs])
    grounding_check = GroundingValidator.is_relevant(request.question, context_combined)
    
    # Extract and validate citations
    citations_found = CitationExtractor.extract_citations(answer)
    citation_validation = CitationExtractor.validate_citations(answer, len(source_docs))
    
    # Determine grounding state
    if not grounding_check["is_relevant"]:
        grounding_state = "rejected"
        confidence = 0.0
    elif "do not contain" in answer.lower() or "insufficient" in answer.lower():
        grounding_state = "rejected"
        confidence = 0.0
    elif citations_found and citation_validation["valid"]:
        grounding_state = "grounded"
        confidence = grounding_check["relevance_score"]
    else:
        grounding_state = "partial"
        confidence = grounding_check["relevance_score"] * 0.5
    
    # Build citation objects
    citations = []
    for i, doc in enumerate(source_docs, 1):
        citations.append(Citation(
            source_id=i,
            text=doc.page_content[:100] + "...",
            relevance_score=doc.metadata.get("relevance_score", 0.0)
        ))
    
    return QueryResponse(
        answer=answer,
        grounding_state=grounding_state,
        citations=citations,
        grounding_confidence=confidence,
        validation_results={
            "citation_validation": citation_validation,
            "grounding_check": grounding_check
        },
        metadata={
            "domain": request.domain,
            "num_sources": len(source_docs),
            "template_used": request.domain
        }
    )

@app.post("/test-prompt")
async def test_prompt(request: PromptTestRequest):
    """Test a specific prompt template with custom context. Returns 200 with message when no API key."""
    if llm is None:
        return {
            "prompt_used": "",
            "response": "Set GEMINI_API_KEY or GOOGLE_API_KEY for test-prompt.",
            "template_name": request.template_name,
            "demo_mode": True
        }
    if request.template_name not in PROMPT_TEMPLATES:
        raise HTTPException(400, f"Invalid template. Choose from: {list(PROMPT_TEMPLATES.keys())}")
    
    template = PROMPT_TEMPLATES[request.template_name]
    prompt = template.format(context=request.context, question=request.question)
    
    response = llm.invoke(prompt)
    
    return {
        "prompt_used": prompt,
        "response": response.content,
        "template_name": request.template_name
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics. Never raises — returns safe defaults on error."""
    try:
        total = len(vector_store.get()["documents"]) if vector_store else 0
    except Exception:
        total = 0
    return {
        "templates_available": list(PROMPT_TEMPLATES.keys()),
        "vector_store_initialized": vector_store is not None,
        "total_documents": total,
        "retrieval_chains": list(retrieval_chains.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
