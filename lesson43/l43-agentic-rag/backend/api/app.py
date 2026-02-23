"""
L43: FastAPI application — Agentic RAG endpoints
"""
from __future__ import annotations
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ..core.models import QueryRequest, SynthesisResult, PipelineState
from ..core.orchestrator import AgenticRAGOrchestrator
from ..core.tracer import AuditStore, PipelineTracer

# API key must be set via env (e.g. by start.sh). Get one at https://aistudio.google.com/apikey
if not os.environ.get("GEMINI_API_KEY"):
    import warnings
    warnings.warn("GEMINI_API_KEY not set — set it before calling /query (e.g. export GEMINI_API_KEY=...)")

app = FastAPI(
    title="L43 Agentic RAG API",
    description="End-to-end Agentic RAG: Planner → Retriever → Validator → Synthesizer",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_orchestrator: AgenticRAGOrchestrator | None = None
_last_query_time: float = 0.0
_QUERY_COOLDOWN_SEC = 4.0  # min seconds between /query to avoid bursting; demo fallback handles 429


def get_orchestrator() -> AgenticRAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgenticRAGOrchestrator()
    return _orchestrator


@app.get("/health")
async def health():
    return {"status": "ok", "lesson": "L43", "pipeline": "Agentic RAG End-to-End"}


@app.post("/query")
async def query(request: QueryRequest):
    global _last_query_time
    now = time.time()
    if now - _last_query_time < _QUERY_COOLDOWN_SEC:
        wait = round(_QUERY_COOLDOWN_SEC - (now - _last_query_time))
        tracer = PipelineTracer()
        msg = f"Please wait {wait} second(s) between queries to avoid rate limits."
        result = SynthesisResult(
            trace_id=tracer.trace_id,
            query=request.query,
            response=msg,
            citations=[],
            tokens_used=0,
            latency_ms=0,
            quality_score=0.0,
            pipeline_state=PipelineState.DONE,
        )
        return result.model_dump(mode="json")
    orch = get_orchestrator()
    result = await orch.run(request)
    _last_query_time = time.time()
    return result.model_dump(mode="json")


@app.get("/audit/{trace_id}")
async def get_audit(trace_id: str):
    record = AuditStore.get(trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found")
    return record


@app.get("/history")
async def get_history(n: int = 10):
    """Returns recent audit records. Consumed by L44 Ragas evaluator."""
    return AuditStore.list_recent(n)


@app.get("/documents/source/{source_id}", response_class=HTMLResponse)
async def get_document_source(
    source_id: str,
    query: str | None = None,
    snippet: str | None = None,
):
    """Serve citation source with optional query context and cited excerpt so the page shows valid information for that query."""
    orch = get_orchestrator()
    try:
        result = orch.retriever._collection.get(
            where={"source": source_id},
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Source not found")
    if not result or not result.get("documents"):
        raise HTTPException(status_code=404, detail="No content for this source")
    docs = result["documents"]
    _escape = lambda t: (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    query_esc = _escape(query) if query else ""
    snippet_esc = _escape(snippet) if snippet else ""
    body_sections = []
    if query_esc:
        body_sections.append(
            f"<section class='context'><h2>Cited in response to</h2><p class='query'>{query_esc}</p></section>"
        )
    if snippet_esc:
        body_sections.append(
            f"<section class='excerpt'><h2>Relevant excerpt</h2><p class='snippet'>{snippet_esc}</p></section>"
        )
    full_parts = [
        f"<section class='chunk'><p>{_escape(doc)}</p></section>" for doc in docs
    ]
    body_sections.append("<h2>Full source content</h2>" + "\n".join(full_parts))
    body_html = "\n".join(body_sections)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_escape(source_id)} — Citation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #e2e8f0; background: #0f1117; line-height: 1.6; }}
    h1 {{ font-size: 1.15rem; color: #63b3ed; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 0.95rem; color: #68d391; margin: 1.25rem 0 0.5rem; }}
    .sub {{ color: #8892a4; font-size: 0.9rem; margin-bottom: 1rem; }}
    section.context {{ background: #161b22; border: 1px solid #2a2f3a; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    section.excerpt {{ background: #1c3c2e; border: 1px solid #68d39144; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    p.query {{ margin: 0; color: #e2e8f0; }}
    p.snippet {{ margin: 0; white-space: pre-wrap; color: #d1d5db; }}
    section.chunk {{ margin-bottom: 1rem; }}
    section.chunk p {{ white-space: pre-wrap; margin: 0; color: #8892a4; }}
    a {{ color: #68d391; }}
  </style>
</head>
<body>
  <h1>Source: {_escape(source_id)}</h1>
  <p class="sub">Knowledge base citation — content used to answer your query.</p>
  {body_html}
  <p style="margin-top:2rem;"><a href="javascript:window.close()">Close</a></p>
</body>
</html>"""
    return html


@app.post("/ingest")
async def ingest_documents(documents: list[str], sources: list[str] | None = None):
    """Seed the ChromaDB collection with documents."""
    orch = get_orchestrator()
    if sources is None:
        sources = [f"doc_{i}" for i in range(len(documents))]
    count = orch.retriever.ingest(documents, sources)
    return {"ingested": count, "collection": orch.retriever._collection_name}
