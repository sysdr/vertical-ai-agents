from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import os
from dotenv import load_dotenv
from backend.models.synthesis_models import (
    SynthesisRequest, SynthesisResponse, Chunk,
    Citation, ReasoningStep, SynthesisState,
)
from backend.agents.synthesizer_agent import SynthesizerAgent
from datetime import datetime

# Load .env from project root (where start.sh runs) so the key is always found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY_PLACEHOLDER = "your_gemini_api_key_here"
GEMINI_API_KEY_RAW = (os.getenv("GEMINI_API_KEY") or "").strip()

def api_key_invalid_response():
    return JSONResponse(
        status_code=503,
        content={
            "detail": "GEMINI_API_KEY invalid or missing. Set a valid key in .env (see https://aistudio.google.com/apikey).",
            "error": "api_key_required",
        },
        headers=CORS_HEADERS,
    )

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

# CORS headers to add to all responses (including errors) so browser never blocks
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
}

app = FastAPI(title="VAIA L41: Synthesizer Agent API")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return JSON error with CORS headers so 500s don't show as CORS errors in the browser."""
    detail = str(exc)
    if "API_KEY" in detail or "API key" in detail or "InvalidArgument" in detail:
        status_code = 503
        message = "GEMINI_API_KEY invalid or missing. Set a valid key in .env (see https://aistudio.google.com/apikey)."
    else:
        status_code = 500
        message = detail
    return JSONResponse(
        status_code=status_code,
        content={"detail": message, "error": "synthesis_failed"},
        headers=CORS_HEADERS,
    )


# CORS middleware (for normal responses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize Synthesizer Agent (only if we have a real API key)
def _valid_api_key():
    key = GEMINI_API_KEY_RAW
    return key and key != API_KEY_PLACEHOLDER

if _valid_api_key():
    synthesizer = SynthesizerAgent(
        api_key=GEMINI_API_KEY_RAW,
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
        citation_granularity=os.getenv("CITATION_GRANULARITY", "medium"),
    )
else:
    synthesizer = None


def _mock_synthesis_response(query: str, chunks: list, synthesis_time_ms: float = 420.0) -> SynthesisResponse:
    """Return a plausible demo response when GEMINI_API_KEY is not set (dashboard still works)."""
    now = datetime.now()
    return SynthesisResponse(
        response_text=f"""Based on the provided context, here is a synthesized answer to: "{query[:80]}..."

The Synthesizer Agent implements explainability in production VAIAs through:
1. **Chain-of-Thought (CoT) reasoning** – decomposing synthesis into explicit steps (context analysis, claim formulation, citation mapping).
2. **Structured XAI** – transparent reasoning patterns; production systems balance transparency and performance with adaptive XAI.
3. **Citation granularity** – from document-level to excerpt-level; legal and medical use cases often require fine-grained citations for compliance.

---
Sources:
[1] VAIA_Architecture_Guide.pdf, §4.2
[2] XAI_Best_Practices.pdf, §2.1
[3] Enterprise_RAG_Patterns.pdf, §3.4

(Demo mode – set GEMINI_API_KEY in .env for live Gemini synthesis.)""",
        citations=[
            Citation(chunk_id=c.id, source=c.source, section=c.section, text_excerpt=c.content[:150] + "...", confidence=c.quality_score or 0.85)
            for c in chunks[:5]
        ],
        reasoning_chain=[
            ReasoningStep(step_name="context_summary", description="Summarize key points from context", input_data={}, output_data={}, confidence=0.88, latency_ms=120.0, timestamp=now),
            ReasoningStep(step_name="claim_formulation", description="Formulate response claims", input_data={}, output_data={}, confidence=0.85, latency_ms=150.0, timestamp=now),
            ReasoningStep(step_name="citation_mapping", description="Map claims to sources", input_data={}, output_data={}, confidence=0.9, latency_ms=150.0, timestamp=now),
        ],
        overall_confidence=0.87,
        synthesis_time_ms=synthesis_time_ms,
        state=SynthesisState.READY_TO_SEND,
        metadata={"mock": True, "message": "Set GEMINI_API_KEY in .env for live Gemini synthesis."},
    )


@app.get("/")
async def root():
    return {
        "service": "VAIA L41: Synthesizer Agent & XAI",
        "status": "operational",
        "components": ["CoTEngine", "CitationMapper", "CoherenceValidator", "XAIFormatter"]
    }

@app.get("/health")
async def health():
    state = synthesizer.state.value if synthesizer else "api_key_missing"
    return {"status": "healthy", "synthesizer_state": state}

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_response(request: SynthesisRequest):
    """Generate synthesis with CoT reasoning and citations"""
    if not synthesizer:
        return _mock_synthesis_response(request.query, request.validated_chunks)
    try:
        result = synthesizer.synthesize(request)
        return result
    except Exception as e:
        detail = str(e)
        if "API_KEY" in detail or "API key" in detail or "InvalidArgument" in detail:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "GEMINI_API_KEY invalid or missing. Set a valid key in .env (see https://aistudio.google.com/apikey).",
                    "error": "api_key_required",
                },
                headers=CORS_HEADERS,
            )
        return JSONResponse(
            status_code=500,
            content={"detail": detail, "error": "synthesis_failed"},
            headers=CORS_HEADERS,
        )

@app.post("/synthesize/demo")
async def synthesize_demo():
    """Demo endpoint with sample data. Returns JSON error with CORS on failure."""
    sample_chunks = [
        Chunk(
            id="chunk_001",
            content="The Synthesizer Agent orchestrates Chain-of-Thought reasoning to generate contextually grounded responses. It decomposes synthesis into explicit steps: context analysis, claim formulation, and citation mapping.",
            source="VAIA_Architecture_Guide.pdf",
            section="§4.2",
            quality_score=0.92
        ),
        Chunk(
            id="chunk_002",
            content="Explainable AI (XAI) in VAIAs makes model reasoning transparent through structured explanation patterns. Production systems balance transparency against performance using adaptive XAI that adjusts detail based on context.",
            source="XAI_Best_Practices.pdf",
            section="§2.1",
            quality_score=0.88
        ),
        Chunk(
            id="chunk_003",
            content="Citation granularity in enterprise RAG systems ranges from coarse (document-level) to fine (excerpt-level). Legal and medical applications demand fine-grained citations for compliance.",
            source="Enterprise_RAG_Patterns.pdf",
            section="§3.4",
            quality_score=0.85
        )
    ]
    
    if not synthesizer:
        demo_request = SynthesisRequest(
            query="How does the Synthesizer Agent implement explainability in production VAIAs?",
            validated_chunks=sample_chunks,
            quality_scores={"overall": 0.88},
            metadata={"from_demo": True},
        )
        return _mock_synthesis_response(demo_request.query, sample_chunks)

    demo_request = SynthesisRequest(
        query="How does the Synthesizer Agent implement explainability in production VAIAs?",
        validated_chunks=sample_chunks,
        quality_scores={"overall": 0.88},
        metadata={"from_demo": True}
    )
    
    try:
        result = synthesizer.synthesize(demo_request, trace_id="demo_trace")
        return result
    except Exception as e:
        detail = str(e)
        if "API_KEY" in detail or "API key" in detail or "InvalidArgument" in detail:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "GEMINI_API_KEY invalid or missing. Set a valid key in .env (see https://aistudio.google.com/apikey).",
                    "error": "api_key_required",
                },
                headers=CORS_HEADERS,
            )
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
