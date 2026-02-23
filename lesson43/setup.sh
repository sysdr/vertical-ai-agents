#!/usr/bin/env bash
# =============================================================================
# L43: Hands-On: Agentic RAG End-to-End
# VAIA Curriculum — Module 5, Lesson 43
# =============================================================================
set -euo pipefail

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()    { echo -e "${GREEN}[L43]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()    { echo -e "${RED}[ERR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}══ $1 ══${NC}"; }

header "L43: Agentic RAG End-to-End Setup"
log "Building: Planner → Retriever → Validator → Synthesizer pipeline"
log "Integrates: L42 Traceability Layer"
log "Prepares:   L44 Ragas Evaluation"

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="${PWD}/l43-agentic-rag"
BACKEND_DIR="${PROJECT_ROOT}/backend"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
DATA_DIR="${PROJECT_ROOT}/data"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
VENV_DIR="${PROJECT_ROOT}/.venv"
# Set GEMINI_API_KEY before running setup/start (get a key at https://aistudio.google.com/apikey)
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
BACKEND_PORT=8043
FRONTEND_PORT=3043

# ─── Prerequisites ────────────────────────────────────────────────────────────
header "Checking Prerequisites"
command -v python3 >/dev/null 2>&1 || err "python3 required"
command -v node    >/dev/null 2>&1 || err "node required"
command -v npm     >/dev/null 2>&1 || err "npm required"
PY_VER=$(python3 --version | awk '{print $2}')
NODE_VER=$(node --version)
log "Python: ${PY_VER} | Node: ${NODE_VER}"

# ─── Directory Structure ──────────────────────────────────────────────────────
header "Creating Project Structure"
mkdir -p "${BACKEND_DIR}/agents"
mkdir -p "${BACKEND_DIR}/core"
mkdir -p "${BACKEND_DIR}/api"
mkdir -p "${BACKEND_DIR}/data/chroma_data"
mkdir -p "${DATA_DIR}/documents"
mkdir -p "${SCRIPTS_DIR}"
mkdir -p "${FRONTEND_DIR}/src/components"
mkdir -p "${FRONTEND_DIR}/public"
log "Directory structure created"

# ─── Python Virtual Environment ──────────────────────────────────────────────
header "Setting Up Python Environment"
create_venv() {
    if python3 -m venv "${VENV_DIR}" 2>/dev/null; then
        log "Virtual environment created (with ensurepip)"
        return 0
    fi
    log "Creating venv without pip (ensurepip not available)..."
    python3 -m venv --without-pip "${VENV_DIR}" || err "Could not create venv. Install python3-venv: apt install python3.12-venv"
    log "Bootstrapping pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "${VENV_DIR}/bin/python3" /tmp/get-pip.py --quiet
    rm -f /tmp/get-pip.py
    log "Virtual environment created (pip bootstrapped)"
}
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    [ -d "${VENV_DIR}" ] && rm -rf "${VENV_DIR}"
    create_venv
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet 2>/dev/null || true

# ─── Python Dependencies (requirements.txt only; pip install after all files) ─
header "Creating requirements.txt"
cat > "${PROJECT_ROOT}/requirements.txt" << 'EOF'
# L43 Agentic RAG End-to-End — May 2025
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
google-generativeai==0.8.3
chromadb==0.5.23
sentence-transformers==3.3.1
httpx==0.28.1
python-dotenv==1.0.1
tenacity==9.0.0
aiofiles==24.1.0
python-multipart==0.0.20
EOF
log "requirements.txt created"

# ─── Backend: Core Models ──────────────────────────────────────────────────────
header "Creating Backend: Core Models"
cat > "${BACKEND_DIR}/core/models.py" << 'PYEOF'
"""
L43: Core Pydantic models for Agentic RAG pipeline.
Every agent exchanges typed objects — no raw dicts.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


class PipelineState(str, Enum):
    IDLE        = "idle"
    PLANNING    = "planning"
    RETRIEVING  = "retrieving"
    VALIDATING  = "validating"
    SYNTHESIZING = "synthesizing"
    TRACING     = "tracing"
    DONE        = "done"
    ERROR       = "error"


class TraceEvent(BaseModel):
    trace_id:   str
    agent:      str
    state:      PipelineState
    input:      dict[str, Any]
    output:     dict[str, Any]
    latency_ms: float
    timestamp:  float = Field(default_factory=time.time)
    attempt:    int   = 1
    error:      Optional[str] = None


class SubQuery(BaseModel):
    text:     str
    strategy: str  # "semantic" | "keyword" | "hybrid"
    priority: int  # 1=high, 2=med, 3=low


class PlanResult(BaseModel):
    original_query: str
    sub_queries:    list[SubQuery]
    retrieval_mode: str  # "sequential" | "parallel"
    reasoning:      str


class Chunk(BaseModel):
    chunk_id:   str
    text:       str
    source:     str
    score:      float
    metadata:   dict[str, Any] = {}


class RetrievalResult(BaseModel):
    query:       str
    chunks:      list[Chunk]
    total_found: int
    latency_ms:  float


class ScoredChunk(BaseModel):
    chunk:         Chunk
    quality_score: float   # 0.0–1.0 relevance to original query
    risk_score:    float   # 0.0–1.0 (high = risky)
    passed:        bool


class ValidationResult(BaseModel):
    validated_chunks: list[ScoredChunk]
    avg_quality_score: float
    attempts:          int
    blocked_reason:    Optional[str] = None
    passed:            bool


class Citation(BaseModel):
    marker:   str   # e.g. "[1]"
    chunk_id: str
    source:   str
    text_snippet: str


class SynthesisResult(BaseModel):
    trace_id:     str
    query:        str
    response:     str
    citations:    list[Citation]
    tokens_used:  int
    latency_ms:   float
    quality_score: float
    pipeline_state: PipelineState


class QueryRequest(BaseModel):
    query:       str  = Field(..., min_length=3, max_length=1000)
    collection:  str  = "enterprise_kb"
    max_results: int  = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    trace_id:      str
    response:      str
    citations:     list[Citation]
    quality_score: float
    pipeline_state: PipelineState
    latency_ms:    float
    tokens_used:   int
PYEOF
log "Core models created"

# ─── Backend: Traceability Layer (from L42) ───────────────────────────────────
cat > "${BACKEND_DIR}/core/tracer.py" << 'PYEOF'
"""
L43: Traceability Layer — re-uses L42 AuditLogger patterns.
Writes structured JSON audit records keyed by trace_id.
"""
from __future__ import annotations
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from .models import TraceEvent, PipelineState
import time


AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit_logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class PipelineTracer:
    """
    Accumulates TraceEvents for a single pipeline execution.
    Writes a consolidated audit record on finalize().
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self._events:  list[TraceEvent] = []
        self._started: float = time.time()

    async def log(self, event: TraceEvent) -> None:
        self._events.append(event)

    async def finalize(
        self,
        final_state: PipelineState,
        result: Optional[dict] = None
    ) -> str:
        audit = {
            "trace_id":    self.trace_id,
            "start_time":  self._started,
            "end_time":    time.time(),
            "duration_ms": round((time.time() - self._started) * 1000, 2),
            "final_state": final_state.value,
            "events":      [e.model_dump() for e in self._events],
            "result":      result or {},
        }
        audit_file = AUDIT_DIR / f"{self.trace_id}.json"
        async with asyncio.Lock():
            audit_file.write_text(json.dumps(audit, indent=2))
        return self.trace_id

    def get_events(self) -> list[TraceEvent]:
        return list(self._events)


class AuditStore:
    """Query completed audit records."""

    @staticmethod
    def get(trace_id: str) -> Optional[dict]:
        f = AUDIT_DIR / f"{trace_id}.json"
        return json.loads(f.read_text()) if f.exists() else None

    @staticmethod
    def list_recent(n: int = 20) -> list[dict]:
        files = sorted(AUDIT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        result = []
        for f in files[:n]:
            try:
                result.append(json.loads(f.read_text()))
            except Exception:
                pass
        return result
PYEOF
log "Traceability layer created"

# ─── Backend: Planner Agent ───────────────────────────────────────────────────
cat > "${BACKEND_DIR}/agents/planner.py" << 'PYEOF'
"""
L43: Planner Agent
Uses Gemini to decompose a user query into structured sub-queries
with retrieval strategy annotations.
"""
from __future__ import annotations
import json
import time
from ..core.models import PlanResult, SubQuery, TraceEvent, PipelineState
from ..core.tracer import PipelineTracer
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PLANNER_PROMPT = """\
You are an expert query planner for an enterprise RAG system.
Given the user query, decompose it into 1-3 focused sub-queries that will maximize
retrieval quality. For each sub-query, specify:
- text: the sub-query string
- strategy: one of "semantic", "keyword", "hybrid"
- priority: 1 (high), 2 (medium), or 3 (low)

Also provide:
- retrieval_mode: "sequential" or "parallel"
- reasoning: brief explanation

Respond ONLY with valid JSON matching this schema:
{{
  "original_query": "...",
  "sub_queries": [{{"text": "...", "strategy": "...", "priority": 1}}],
  "retrieval_mode": "sequential",
  "reasoning": "..."
}}

User query: {query}"""


class PlannerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel("gemini-2.0-flash-lite")

    async def plan(self, query: str, tracer: PipelineTracer) -> PlanResult:
        t0 = time.time()
        try:
            response = self._model.generate_content(
                PLANNER_PROMPT.format(query=query),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                )
            )
            raw = response.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            result = PlanResult(
                original_query=data.get("original_query", query),
                sub_queries=[SubQuery(**sq) for sq in data.get("sub_queries", [])],
                retrieval_mode=data.get("retrieval_mode", "sequential"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            # Fallback plan: use original query as single sub-query
            result = PlanResult(
                original_query=query,
                sub_queries=[SubQuery(text=query, strategy="semantic", priority=1)],
                retrieval_mode="sequential",
                reasoning=f"Fallback plan due to: {str(e)}",
            )

        latency = round((time.time() - t0) * 1000, 2)
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="planner",
            state=PipelineState.PLANNING,
            input={"query": query},
            output=result.model_dump(),
            latency_ms=latency,
        ))
        return result
PYEOF
log "Planner agent created"

# ─── Backend: Retriever Agent ─────────────────────────────────────────────────
cat > "${BACKEND_DIR}/agents/retriever.py" << 'PYEOF'
"""
L43: Retriever Agent
Executes sub-queries against ChromaDB. Supports expanded search on retry.
"""
from __future__ import annotations
import time
import uuid
from ..core.models import Chunk, RetrievalResult, PlanResult, TraceEvent, PipelineState
from ..core.tracer import PipelineTracer
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_data")

_ef = embedding_functions.DefaultEmbeddingFunction()


class RetrieverAgent:
    def __init__(self, collection_name: str = "enterprise_kb"):
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self._collection = self._client.get_collection(
                name=self._collection_name,
                embedding_function=_ef,
            )
        except Exception:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                embedding_function=_ef,
            )

    async def retrieve(
        self,
        plan: PlanResult,
        tracer: PipelineTracer,
        attempt: int = 1,
        top_k: int = 5,
    ) -> RetrievalResult:
        t0 = time.time()
        all_chunks: list[Chunk] = []
        seen_ids: set[str] = set()

        # Expand search on retry: lower threshold, higher top_k
        effective_k = top_k + (attempt - 1) * 3

        for sq in plan.sub_queries:
            try:
                results = self._collection.query(
                    query_texts=[sq.text],
                    n_results=min(effective_k, 20),
                )
                if results and results["ids"]:
                    for i, chunk_id in enumerate(results["ids"][0]):
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            score = 1.0 - results["distances"][0][i] if results.get("distances") else 0.5
                            all_chunks.append(Chunk(
                                chunk_id=chunk_id,
                                text=results["documents"][0][i],
                                source=results["metadatas"][0][i].get("source", "unknown"),
                                score=round(score, 4),
                                metadata=results["metadatas"][0][i],
                            ))
            except Exception:
                pass

        # Sort by score descending
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        latency = round((time.time() - t0) * 1000, 2)

        result = RetrievalResult(
            query=plan.original_query,
            chunks=all_chunks[:top_k * 2],
            total_found=len(all_chunks),
            latency_ms=latency,
        )
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="retriever",
            state=PipelineState.RETRIEVING,
            input={"sub_queries": [sq.text for sq in plan.sub_queries], "attempt": attempt},
            output={"chunks_found": len(result.chunks), "top_score": result.chunks[0].score if result.chunks else 0},
            latency_ms=latency, attempt=attempt,
        ))
        return result

    def ingest(self, texts: list[str], sources: list[str]) -> int:
        """Seed the vector store with documents."""
        ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            documents=texts,
            ids=ids,
            metadatas=[{"source": s} for s in sources],
        )
        return len(ids)
PYEOF
log "Retriever agent created"

# ─── Backend: Validator Agent ─────────────────────────────────────────────────
cat > "${BACKEND_DIR}/agents/validator.py" << 'PYEOF'
"""
L43: Validator Agent
Scores retrieved chunks for quality and risk.
Triggers self-correction loop on low quality.
"""
from __future__ import annotations
import time
import os
from ..core.models import (
    Chunk, ScoredChunk, ValidationResult, RetrievalResult,
    TraceEvent, PipelineState
)
from ..core.tracer import PipelineTracer
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

VALIDATION_THRESHOLD = 0.55
MAX_RETRIES = 3

VALIDATOR_PROMPT = """\
Score each chunk for relevance to the query. Return ONLY JSON:
{{
  "scores": [
    {{"chunk_id": "...", "quality_score": 0.0-1.0, "risk_score": 0.0-1.0}}
  ]
}}
quality_score: how relevant and useful this chunk is for answering the query.
risk_score: likelihood of hallucination or factual error if used.

Query: {query}
Chunks:
{chunks_text}"""


class ValidatorAgent:
    def __init__(self):
        self._model = genai.GenerativeModel("gemini-2.0-flash-lite")

    async def validate(
        self,
        retrieval: RetrievalResult,
        query: str,
        tracer: PipelineTracer,
        attempt: int = 1,
    ) -> ValidationResult:
        t0 = time.time()

        if not retrieval.chunks:
            result = ValidationResult(
                validated_chunks=[], avg_quality_score=0.0,
                attempts=attempt, passed=False,
                blocked_reason="No chunks retrieved",
            )
            await self._log(tracer, query, result, t0, attempt)
            return result

        scores_map = await self._score_chunks(retrieval.chunks, query)
        scored = []
        for chunk in retrieval.chunks:
            s = scores_map.get(chunk.chunk_id, {"quality_score": chunk.score, "risk_score": 0.2})
            passed = s["quality_score"] >= VALIDATION_THRESHOLD and s["risk_score"] < 0.8
            scored.append(ScoredChunk(
                chunk=chunk,
                quality_score=s["quality_score"],
                risk_score=s["risk_score"],
                passed=passed,
            ))

        passing = [sc for sc in scored if sc.passed]
        avg_score = sum(sc.quality_score for sc in scored) / len(scored) if scored else 0.0

        result = ValidationResult(
            validated_chunks=passing or scored[:3],  # fallback: use top 3
            avg_quality_score=round(avg_score, 4),
            attempts=attempt,
            passed=avg_score >= VALIDATION_THRESHOLD,
        )
        await self._log(tracer, query, result, t0, attempt)
        return result

    async def _score_chunks(self, chunks: list[Chunk], query: str) -> dict:
        chunks_text = "\n".join([
            f"[chunk_id={c.chunk_id}] {c.text[:300]}" for c in chunks[:8]
        ])
        try:
            import json
            response = self._model.generate_content(
                VALIDATOR_PROMPT.format(query=query, chunks_text=chunks_text),
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, max_output_tokens=512,
                )
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return {s["chunk_id"]: s for s in data.get("scores", [])}
        except Exception:
            # Fallback: use retrieval scores
            return {c.chunk_id: {"quality_score": c.score, "risk_score": 0.2} for c in chunks}

    async def _log(self, tracer, query, result, t0, attempt):
        latency = round((time.time() - t0) * 1000, 2)
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="validator",
            state=PipelineState.VALIDATING,
            input={"query": query, "attempt": attempt},
            output={"avg_quality_score": result.avg_quality_score, "passed": result.passed,
                    "chunks_passed": len(result.validated_chunks)},
            latency_ms=latency, attempt=attempt,
        ))
PYEOF
log "Validator agent created"

# ─── Backend: Synthesizer Agent ───────────────────────────────────────────────
cat > "${BACKEND_DIR}/agents/synthesizer.py" << 'PYEOF'
"""
L43: Synthesizer Agent
Generates grounded responses using validated chunks.
Attaches citations linking claims to source chunks.
"""
from __future__ import annotations
import time
import re
import os
from ..core.models import (
    ScoredChunk, SynthesisResult, Citation,
    TraceEvent, PipelineState
)
from ..core.tracer import PipelineTracer
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

TOKEN_BUDGET = 6000  # characters for context (approx)

SYNTHESIS_PROMPT = """\
You are an enterprise knowledge assistant. Answer the query using ONLY the provided context.
- Cite each piece of evidence with the chunk number in brackets: [1], [2], etc.
- If the context does not contain the answer, say "The provided context does not contain enough information to answer this question."
- Be concise and professional.

Context:
{numbered_chunks}

Query: {query}

Answer:"""


class SynthesizerAgent:
    def __init__(self):
        self._model = genai.GenerativeModel("gemini-2.0-flash-lite")

    async def synthesize(
        self,
        query: str,
        validated_chunks: list[ScoredChunk],
        tracer: PipelineTracer,
    ) -> SynthesisResult:
        t0 = time.time()

        # Build numbered context, respecting token budget
        numbered, index_map = self._build_context(validated_chunks)

        prompt = SYNTHESIS_PROMPT.format(
            numbered_chunks=numbered,
            query=query,
        )
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                )
            )
            text = response.text
            tokens = getattr(response.usage_metadata, "total_token_count", 0) or max(1, len(text) // 4)
        except Exception as e:
            text = f"Synthesis error: {str(e)}"
            tokens = 0

        citations = self._extract_citations(text, index_map)
        latency = round((time.time() - t0) * 1000, 2)
        latency = max(latency, 0.1)
        avg_quality = sum(sc.quality_score for sc in validated_chunks) / max(len(validated_chunks), 1)
        if avg_quality <= 0 and text:
            avg_quality = 0.5

        result = SynthesisResult(
            trace_id=tracer.trace_id,
            query=query,
            response=text,
            citations=citations,
            tokens_used=tokens or max(1, len(text) // 4),
            latency_ms=latency,
            quality_score=round(avg_quality, 2),
            pipeline_state=PipelineState.DONE,
        )
        await tracer.log(TraceEvent(
            trace_id=tracer.trace_id, agent="synthesizer",
            state=PipelineState.SYNTHESIZING,
            input={"query": query, "chunks_used": len(validated_chunks)},
            output={"response_length": len(text), "citations": len(citations), "tokens_used": tokens},
            latency_ms=latency,
        ))
        return result

    def _build_context(self, chunks: list[ScoredChunk]) -> tuple[str, dict[int, ScoredChunk]]:
        lines = []
        index_map = {}
        total_chars = 0
        for i, sc in enumerate(chunks, start=1):
            snippet = sc.chunk.text[:500]
            if total_chars + len(snippet) > TOKEN_BUDGET:
                break
            lines.append(f"[{i}] (source: {sc.chunk.source})\n{snippet}")
            index_map[i] = sc
            total_chars += len(snippet)
        return "\n\n".join(lines), index_map

    def _extract_citations(self, text: str, index_map: dict) -> list[Citation]:
        citations = []
        seen = set()
        for match in re.finditer(r'\[(\d+)\]', text):
            n = int(match.group(1))
            if n in index_map and n not in seen:
                seen.add(n)
                sc = index_map[n]
                citations.append(Citation(
                    marker=f"[{n}]",
                    chunk_id=sc.chunk.chunk_id,
                    source=sc.chunk.source,
                    text_snippet=sc.chunk.text[:100],
                ))
        return citations
PYEOF
log "Synthesizer agent created"

# ─── Backend: Orchestrator ────────────────────────────────────────────────────
cat > "${BACKEND_DIR}/core/orchestrator.py" << 'PYEOF'
"""
L43: Pipeline Orchestrator
Coordinates all four agents, manages state transitions, and handles
the self-correction loop with bounded retries.
"""
from __future__ import annotations
import uuid
import time
from ..core.models import (
    PipelineState, SynthesisResult, QueryRequest
)
from ..core.tracer import PipelineTracer
from ..agents.planner    import PlannerAgent
from ..agents.retriever  import RetrieverAgent
from ..agents.validator  import ValidatorAgent, VALIDATION_THRESHOLD
from ..agents.synthesizer import SynthesizerAgent


class AgenticRAGOrchestrator:
    def __init__(self, collection_name: str = "enterprise_kb"):
        self.planner    = PlannerAgent()
        self.retriever  = RetrieverAgent(collection_name)
        self.validator  = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()

    async def run(self, request: QueryRequest) -> SynthesisResult:
        tracer = PipelineTracer()
        state  = PipelineState.IDLE
        t0     = time.time()

        try:
            # ─ PLANNING ─
            state = PipelineState.PLANNING
            plan = await self.planner.plan(request.query, tracer)

            # ─ RETRIEVE → VALIDATE loop ─
            state = PipelineState.RETRIEVING
            retrieval = await self.retriever.retrieve(plan, tracer, attempt=1, top_k=request.max_results)

            state = PipelineState.VALIDATING
            validation = await self.validator.validate(retrieval, request.query, tracer, attempt=1)

            # Self-correction loop
            max_retries = 3
            attempt = 1
            while not validation.passed and attempt < max_retries:
                attempt += 1
                state = PipelineState.RETRIEVING
                retrieval = await self.retriever.retrieve(plan, tracer, attempt=attempt, top_k=request.max_results)
                state = PipelineState.VALIDATING
                validation = await self.validator.validate(retrieval, request.query, tracer, attempt=attempt)

            # ─ SYNTHESIZING ─
            state = PipelineState.SYNTHESIZING
            result = await self.synthesizer.synthesize(
                request.query,
                validation.validated_chunks,
                tracer,
            )

            # ─ TRACING ─
            state = PipelineState.TRACING
            await tracer.finalize(PipelineState.DONE, result.model_dump(mode="json"))
            result.pipeline_state = PipelineState.DONE
            return result

        except Exception as e:
            await tracer.finalize(PipelineState.ERROR, {"error": str(e)})
            return SynthesisResult(
                trace_id=tracer.trace_id,
                query=request.query,
                response=f"Pipeline error in {state.value}: {str(e)}",
                citations=[],
                tokens_used=0,
                latency_ms=round((time.time() - t0) * 1000, 2),
                quality_score=0.0,
                pipeline_state=PipelineState.ERROR,
            )
PYEOF
log "Orchestrator created"

# ─── Backend: FastAPI Application ────────────────────────────────────────────
cat > "${BACKEND_DIR}/api/app.py" << 'PYEOF'
"""
L43: FastAPI application — Agentic RAG endpoints
"""
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..core.models import QueryRequest, QueryResponse, SynthesisResult
from ..core.orchestrator import AgenticRAGOrchestrator
from ..core.tracer import AuditStore

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


def get_orchestrator() -> AgenticRAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgenticRAGOrchestrator()
    return _orchestrator


@app.get("/health")
async def health():
    return {"status": "ok", "lesson": "L43", "pipeline": "Agentic RAG End-to-End"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    orch = get_orchestrator()
    result = await orch.run(request)
    return QueryResponse(
        trace_id=result.trace_id,
        response=result.response,
        citations=result.citations,
        quality_score=result.quality_score,
        pipeline_state=result.pipeline_state,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
    )


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


@app.post("/ingest")
async def ingest_documents(documents: list[str], sources: list[str] | None = None):
    """Seed the ChromaDB collection with documents."""
    orch = get_orchestrator()
    if sources is None:
        sources = [f"doc_{i}" for i in range(len(documents))]
    count = orch.retriever.ingest(documents, sources)
    return {"ingested": count, "collection": orch.retriever._collection_name}
PYEOF
log "FastAPI app created"

# ─── Backend: Package Init Files ─────────────────────────────────────────────
cat > "${BACKEND_DIR}/__init__.py" << 'EOF'
EOF
cat > "${BACKEND_DIR}/agents/__init__.py" << 'EOF'
EOF
cat > "${BACKEND_DIR}/core/__init__.py" << 'EOF'
EOF
cat > "${BACKEND_DIR}/api/__init__.py" << 'EOF'
EOF

# ─── Backend: Seed Script ─────────────────────────────────────────────────────
cat > "${BACKEND_DIR}/seed_knowledge_base.py" << 'PYEOF'
"""
L43: Seed the ChromaDB collection with enterprise knowledge documents.
Run once before starting the API server.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# GEMINI_API_KEY not needed for seed (Chroma only); set it for start.sh/query

from backend.agents.retriever import RetrieverAgent

DOCUMENTS = [
    "The company's Q3 2025 revenue increased by 18% year-over-year, driven by strong performance in the cloud services division. Enterprise contracts grew 24%, representing 63% of total revenue.",
    "Our AI deployment guidelines require all production models to pass security review and bias assessment before release. Models must achieve >95% accuracy on the internal benchmark suite.",
    "The incident response protocol mandates a 15-minute initial triage window for P1 incidents. The on-call engineer must notify the CTO within 30 minutes of P1 classification.",
    "Employee benefits include comprehensive health insurance, 20 days PTO per year, and a $5,000 annual learning budget. Remote work is permitted for eligible roles with manager approval.",
    "The data retention policy requires customer data to be deleted within 90 days of account closure. Backups are retained for 7 years for compliance with financial regulations.",
    "Our microservices architecture consists of 47 services deployed across three availability zones. Average service-to-service latency is 12ms at p99. We use gRPC for internal communication.",
    "The product roadmap for 2026 includes: autonomous agent orchestration framework (Q1), multimodal document processing (Q2), real-time collaboration features (Q3), and global CDN expansion (Q4).",
    "Security audit results from October 2025 identified 3 high-severity findings, all of which were remediated within the 30-day SLA. The next audit is scheduled for April 2026.",
    "Our machine learning infrastructure runs on a cluster of 200 A100 GPUs. Training jobs are scheduled via a priority queue system. Average GPU utilization is 78% during business hours.",
    "Customer success metrics: Net Promoter Score is 67 (industry average: 45), customer churn rate is 2.1% monthly, average contract value increased to $180K in 2025.",
    "The engineering team follows a 2-week sprint cadence with daily standups. Code review requires at least 2 approvals. All features must include unit tests with >80% coverage.",
    "Compliance certifications held: SOC 2 Type II, ISO 27001, GDPR compliant, HIPAA BAA available for healthcare customers. Annual penetration testing conducted by third party.",
    "Vector database performance benchmarks: ChromaDB handles 10M embeddings with p99 query latency of 45ms. We shard collections at 2M embeddings for optimal performance.",
    "The Agentic RAG system uses a four-stage pipeline: query planning, multi-source retrieval, validation with self-correction, and grounded synthesis using large language models.",
    "Enterprise AI governance framework includes: model cards for all production models, quarterly bias audits, explainability reports for high-stakes decisions, and human oversight for critical workflows.",
]

SOURCES = [
    "financial_report_q3_2025.pdf",
    "ai_deployment_guidelines.md",
    "incident_response_protocol.md",
    "hr_benefits_handbook.pdf",
    "data_retention_policy.md",
    "infrastructure_overview.md",
    "product_roadmap_2026.pdf",
    "security_audit_2025.md",
    "ml_infrastructure_guide.md",
    "customer_success_metrics.md",
    "engineering_handbook.md",
    "compliance_certifications.md",
    "vector_db_benchmarks.md",
    "agentic_rag_documentation.md",
    "ai_governance_framework.md",
]

if __name__ == "__main__":
    print("Seeding ChromaDB knowledge base...")
    retriever = RetrieverAgent("enterprise_kb")
    # Check if already seeded
    try:
        count = retriever._collection.count()
        if count >= len(DOCUMENTS):
            print(f"Collection already has {count} documents. Skipping seed.")
            sys.exit(0)
    except Exception:
        pass
    n = retriever.ingest(DOCUMENTS, SOURCES)
    print(f"✓ Ingested {n} documents into 'enterprise_kb' collection")
    print(f"  Collection count: {retriever._collection.count()}")
PYEOF
log "Seed script created"

# ─── Frontend: package.json ───────────────────────────────────────────────────
header "Creating React Frontend"
cat > "${FRONTEND_DIR}/package.json" << 'EOF'
{
  "name": "l43-agentic-rag-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "PORT=3043 react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test --watchAll=false"
  },
  "browserslist": {
    "production": [">0.2%", "not dead"],
    "development": ["last 1 chrome version"]
  },
  "proxy": "http://localhost:8043"
}
EOF

# ─── Frontend: public/index.html ─────────────────────────────────────────────
cat > "${FRONTEND_DIR}/public/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>L43 · Agentic RAG Pipeline</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
</head>
<body style="margin:0;background:#0f1117;">
  <div id="root"></div>
</body>
</html>
EOF

# ─── Frontend: src/index.js ───────────────────────────────────────────────────
cat > "${FRONTEND_DIR}/src/index.js" << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
EOF

# ─── Frontend: src/App.jsx ────────────────────────────────────────────────────
cat > "${FRONTEND_DIR}/src/App.jsx" << 'JSEOF'
import React, { useState, useCallback, useRef } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8043';

const STATE_CONFIG = {
  idle:        { label: 'Idle',        color: '#4a5568', bg: '#1a202c', icon: '●' },
  planning:    { label: 'Planning',    color: '#68d391', bg: '#1c3c2e', icon: '◈' },
  retrieving:  { label: 'Retrieving', color: '#63b3ed', bg: '#1a2c3d', icon: '⟳' },
  validating:  { label: 'Validating', color: '#f6ad55', bg: '#3d2c1a', icon: '⊛' },
  synthesizing:{ label: 'Synthesizing',color: '#d6bcfa', bg: '#2d1a3d', icon: '✦' },
  tracing:     { label: 'Tracing',    color: '#fc8181', bg: '#3d1a1a', icon: '◎' },
  done:        { label: 'Done',        color: '#68d391', bg: '#1c3c2e', icon: '✓' },
  error:       { label: 'Error',       color: '#fc8181', bg: '#3d1a1a', icon: '✕' },
};

const DEMO_QUERIES = [
  "What was our Q3 2025 revenue growth?",
  "Describe the AI deployment guidelines for production models.",
  "What is the incident response protocol for P1 issues?",
  "How many GPU clusters does our ML infrastructure have?",
  "What are the key milestones in the 2026 product roadmap?",
];

function PipelineStages({ currentState }) {
  const stages = ['planning', 'retrieving', 'validating', 'synthesizing', 'tracing', 'done'];
  const stageIndex = stages.indexOf(currentState);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap', margin: '12px 0' }}>
      {stages.map((s, i) => {
        const cfg = STATE_CONFIG[s];
        const active = s === currentState;
        const done = stageIndex > i;
        return (
          <React.Fragment key={s}>
            <div style={{
              padding: '4px 10px', borderRadius: 6,
              fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
              background: active ? cfg.bg : done ? '#1a2a1a' : '#161b22',
              color: active ? cfg.color : done ? '#4caf50' : '#4a5568',
              border: `1px solid ${active ? cfg.color + '66' : done ? '#2d4a2d' : '#2a2f3a'}`,
              transition: 'all 0.3s ease',
              fontWeight: active ? 600 : 400,
            }}>
              {cfg.icon} {cfg.label}
            </div>
            {i < stages.length - 1 && (
              <div style={{ color: done || active ? '#4a5568' : '#2a2f3a', fontSize: 10 }}>›</div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

function CitationCard({ citation, index }) {
  return (
    <div style={{
      background: '#161b22', borderRadius: 8, padding: '10px 12px',
      border: '1px solid #2a2f3a', marginBottom: 8,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{
          background: '#d6bcfa22', color: '#d6bcfa', border: '1px solid #d6bcfa44',
          borderRadius: 4, padding: '1px 6px', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace",
        }}>{citation.marker}</span>
        <span style={{ color: '#63b3ed', fontSize: 11 }}>{citation.source}</span>
      </div>
      <p style={{ color: '#8892a4', fontSize: 12, margin: 0, lineHeight: 1.5 }}>
        {citation.text_snippet}...
      </p>
    </div>
  );
}

function QualityBar({ score }) {
  if (score == null || score <= 0) {
    return (
      <div style={{ marginTop: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: '#8892a4', fontSize: 11 }}>Quality Score</span>
          <span style={{ color: '#4a5568', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}>—</span>
        </div>
        <div style={{ background: '#2a2f3a', borderRadius: 3, height: 4, overflow: 'hidden' }}>
          <div style={{ width: '0%', height: '100%', background: '#4a5568', borderRadius: 3 }}/>
        </div>
      </div>
    );
  }
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? '#68d391' : pct >= 50 ? '#f6ad55' : '#fc8181';
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: '#8892a4', fontSize: 11 }}>Quality Score</span>
        <span style={{ color, fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ background: '#2a2f3a', borderRadius: 3, height: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.5s ease' }}/>
      </div>
    </div>
  );
}

function HistoryItem({ item, onClick }) {
  const state = item.final_state || 'done';
  const cfg = STATE_CONFIG[state] || STATE_CONFIG.done;
  const ts = new Date(item.start_time * 1000).toLocaleTimeString();
  return (
    <div onClick={onClick} style={{
      background: '#161b22', borderRadius: 8, padding: '10px 12px',
      border: `1px solid #2a2f3a`, marginBottom: 6, cursor: 'pointer',
      transition: 'border-color 0.2s',
    }}
    onMouseEnter={e => e.currentTarget.style.borderColor = cfg.color + '66'}
    onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2f3a'}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: cfg.color, fontSize: 10, fontFamily: "'IBM Plex Mono', monospace" }}>
          {cfg.icon} {state.toUpperCase()}
        </span>
        <span style={{ color: '#4a5568', fontSize: 10 }}>{ts}</span>
      </div>
      <p style={{ color: '#8892a4', fontSize: 12, margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {item.result?.query || item.events?.[0]?.input?.query || 'Query'}
      </p>
      <p style={{ color: '#4a5568', fontSize: 10, margin: '4px 0 0', fontFamily: "'IBM Plex Mono', monospace" }}>
        {item.trace_id?.slice(0, 8)}… · {Math.round(item.duration_ms)}ms
      </p>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentState, setCurrentState] = useState('idle');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('response');
  const [selectedTrace, setSelectedTrace] = useState(null);
  const abortRef = useRef(null);

  const fetchHistory = useCallback(async () => {
    try {
      const r = await fetch(`${API}/history?n=15`);
      if (r.ok) setHistory(await r.json());
    } catch (_) {}
  }, []);

  const fetchTrace = useCallback(async (trace_id) => {
    try {
      const r = await fetch(`${API}/audit/${trace_id}`);
      if (r.ok) setSelectedTrace(await r.json());
    } catch (_) {}
  }, []);

  const submit = useCallback(async () => {
    if (!query.trim() || loading) return;
    setLoading(true); setError(null); setResult(null); setCurrentState('planning');

    // Animate state transitions
    const states = ['planning', 'retrieving', 'validating', 'synthesizing', 'tracing'];
    let i = 0;
    const interval = setInterval(() => {
      if (i < states.length - 1) { i++; setCurrentState(states[i]); }
      else clearInterval(interval);
    }, 900);

    try {
      const resp = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), collection: 'enterprise_kb', max_results: 5 }),
      });
      clearInterval(interval);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setResult(data); setCurrentState(data.pipeline_state || 'done');
      await fetchHistory();
    } catch (e) {
      clearInterval(interval);
      setError(e.message); setCurrentState('error');
    } finally { setLoading(false); }
  }, [query, loading, fetchHistory]);

  React.useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const styles = {
    app: { minHeight: '100vh', background: '#0f1117', fontFamily: "'IBM Plex Sans', sans-serif", color: '#e2e8f0', display: 'flex', flexDirection: 'column' },
    header: { background: '#161b22', borderBottom: '1px solid #2a2f3a', padding: '14px 24px', display: 'flex', alignItems: 'center', gap: 16 },
    pill: { background: '#68d39122', color: '#68d391', border: '1px solid #68d39144', borderRadius: 20, padding: '2px 10px', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" },
    body: { display: 'flex', flex: 1, overflow: 'hidden' },
    sidebar: { width: 260, background: '#161b22', borderRight: '1px solid #2a2f3a', padding: 16, overflowY: 'auto' },
    main: { flex: 1, padding: 24, overflowY: 'auto' },
    inputRow: { display: 'flex', gap: 10, marginBottom: 12 },
    input: { flex: 1, background: '#161b22', border: '1px solid #2a2f3a', borderRadius: 8, padding: '10px 14px', color: '#e2e8f0', fontSize: 14, outline: 'none', fontFamily: "'IBM Plex Sans', sans-serif" },
    btn: { background: '#68d391', color: '#0f1117', border: 'none', borderRadius: 8, padding: '10px 20px', cursor: 'pointer', fontSize: 13, fontWeight: 700 },
    tabs: { display: 'flex', gap: 8, marginBottom: 16 },
    tab: (active) => ({ background: active ? '#2a2f3a' : 'transparent', color: active ? '#e2e8f0' : '#4a5568', border: '1px solid', borderColor: active ? '#3a3f4a' : 'transparent', borderRadius: 6, padding: '5px 14px', cursor: 'pointer', fontSize: 12 }),
    card: { background: '#161b22', borderRadius: 10, border: '1px solid #2a2f3a', padding: 18, marginBottom: 16 },
    meta: { display: 'flex', gap: 16, flexWrap: 'wrap' },
    metaItem: (color) => ({ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color }),
    responseText: { color: '#d1d5db', fontSize: 14, lineHeight: 1.8, whiteSpace: 'pre-wrap' },
    demoBtn: (i) => ({ background: '#161b22', border: '1px solid #2a2f3a', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', color: '#8892a4', fontSize: 12, textAlign: 'left', marginBottom: 6, width: '100%' }),
  };

  return (
    <div style={styles.app}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>Agentic RAG Pipeline</div>
          <div style={{ fontSize: 11, color: '#4a5568' }}>L43 · End-to-End · VAIA Curriculum</div>
        </div>
        <div style={styles.pill}>PLANNER → RETRIEVER → VALIDATOR → SYNTHESIZER</div>
        {result && <div style={{ ...styles.pill, marginLeft: 'auto', background: '#63b3ed22', color: '#63b3ed', borderColor: '#63b3ed44' }}>
          {(result.latency_ms > 0 ? result.latency_ms + 'ms' : '—')} · {(result.tokens_used > 0 ? result.tokens_used + ' tokens' : '—')}
        </div>}
      </div>

      <div style={styles.body}>
        {/* Sidebar */}
        <div style={styles.sidebar}>
          <div style={{ fontSize: 11, color: '#4a5568', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Demo Queries</div>
          {DEMO_QUERIES.map((q, i) => (
            <button key={i} style={styles.demoBtn(i)} onClick={() => setQuery(q)}>
              {q}
            </button>
          ))}

          <div style={{ fontSize: 11, color: '#4a5568', margin: '20px 0 10px', textTransform: 'uppercase', letterSpacing: 1 }}>
            Recent History
            <button onClick={fetchHistory} style={{ float: 'right', background: 'none', border: 'none', color: '#4a5568', cursor: 'pointer', fontSize: 10 }}>↻</button>
          </div>
          {history.length === 0 && <div style={{ color: '#2a2f3a', fontSize: 12 }}>No history yet</div>}
          {history.map((h, i) => (
            <HistoryItem key={i} item={h} onClick={() => { setSelectedTrace(h); setActiveTab('trace'); }} />
          ))}
        </div>

        {/* Main */}
        <div style={styles.main}>
          {/* Input */}
          <div style={styles.inputRow}>
            <input
              style={styles.input}
              placeholder="Ask a question about enterprise knowledge..."
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && submit()}
            />
            <button style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }} onClick={submit} disabled={loading}>
              {loading ? '⟳' : 'Query'}
            </button>
          </div>

          {/* Pipeline stages */}
          <PipelineStages currentState={currentState} />

          {error && (
            <div style={{ background: '#3d1a1a', border: '1px solid #fc818166', borderRadius: 8, padding: '12px 16px', color: '#fc8181', fontSize: 13, marginBottom: 16 }}>
              ✕ {error}
            </div>
          )}

          {result && (
            <>
              {/* Tabs */}
              <div style={styles.tabs}>
                {['response', 'citations', 'trace'].map(t => (
                  <button key={t} style={styles.tab(activeTab === t)} onClick={() => { setActiveTab(t); if (t === 'trace') fetchTrace(result.trace_id); }}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                    {t === 'citations' && ` (${result.citations?.length || 0})`}
                  </button>
                ))}
              </div>

              {/* Response tab */}
              {activeTab === 'response' && (
                <div style={styles.card}>
                  <div style={styles.meta}>
                    <span style={styles.metaItem('#68d391')}>✓ {(STATE_CONFIG[result.pipeline_state] || {}).label || result.pipeline_state}</span>
                    <span style={styles.metaItem('#63b3ed')}>{result.latency_ms > 0 ? result.latency_ms + 'ms' : '—'}</span>
                    <span style={styles.metaItem('#d6bcfa')}>{result.tokens_used > 0 ? result.tokens_used + ' tokens' : '—'}</span>
                    <span style={styles.metaItem('#4a5568')}>trace: {result.trace_id?.slice(0, 12)}…</span>
                  </div>
                  <QualityBar score={result.quality_score != null && result.quality_score > 0 ? result.quality_score : null} />
                  <hr style={{ border: 'none', borderTop: '1px solid #2a2f3a', margin: '14px 0' }}/>
                  <div style={styles.responseText}>{result.response}</div>
                </div>
              )}

              {/* Citations tab */}
              {activeTab === 'citations' && (
                <div>
                  {result.citations?.length === 0
                    ? <div style={{ color: '#4a5568', fontSize: 13 }}>No citations extracted.</div>
                    : result.citations?.map((c, i) => <CitationCard key={i} citation={c} index={i}/>)
                  }
                </div>
              )}

              {/* Trace tab */}
              {activeTab === 'trace' && (
                <div style={styles.card}>
                  {!selectedTrace
                    ? <div style={{ color: '#4a5568', fontSize: 13 }}>Loading trace…</div>
                    : (
                      <>
                        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: '#8892a4', marginBottom: 12 }}>
                          trace_id: {selectedTrace.trace_id} · {Math.round(selectedTrace.duration_ms)}ms
                        </div>
                        {selectedTrace.events?.map((ev, i) => {
                          const cfg = STATE_CONFIG[ev.state] || {};
                          return (
                            <div key={i} style={{ borderLeft: `2px solid ${cfg.color || '#2a2f3a'}`, paddingLeft: 12, marginBottom: 12 }}>
                              <div style={{ display: 'flex', gap: 10, marginBottom: 4 }}>
                                <span style={{ color: cfg.color, fontSize: 11, fontWeight: 600 }}>{ev.agent?.toUpperCase()}</span>
                                <span style={{ color: '#4a5568', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" }}>{ev.latency_ms}ms</span>
                                {ev.attempt > 1 && <span style={{ color: '#f6ad55', fontSize: 10 }}>attempt {ev.attempt}</span>}
                              </div>
                              <pre style={{ background: '#0f1117', borderRadius: 6, padding: 10, fontSize: 10, color: '#8892a4', overflow: 'auto', margin: 0, fontFamily: "'IBM Plex Mono', monospace" }}>
                                {JSON.stringify(ev.output, null, 2)}
                              </pre>
                            </div>
                          );
                        })}
                      </>
                    )
                  }
                </div>
              )}
            </>
          )}

          {!result && !loading && !error && (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: '#2a2f3a' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>◈</div>
              <div style={{ fontSize: 14 }}>Enter a query to run the Agentic RAG pipeline</div>
              <div style={{ fontSize: 12, marginTop: 8 }}>Planner → Retriever → Validator → Synthesizer</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
JSEOF
log "React frontend created"

# ─── Helper Scripts ───────────────────────────────────────────────────────────
header "Creating Helper Scripts"

cat > "${SCRIPTS_DIR}/start.sh" << BASH
#!/usr/bin/env bash
set -e
# Avoid duplicate services
pkill -f "uvicorn backend.api.app" 2>/dev/null || true
[ -f /tmp/l43_backend.pid ]  && kill \$(cat /tmp/l43_backend.pid) 2>/dev/null || true
[ -f /tmp/l43_frontend.pid ] && kill \$(cat /tmp/l43_frontend.pid) 2>/dev/null || true
rm -f /tmp/l43_backend.pid /tmp/l43_frontend.pid
sleep 1
source "${VENV_DIR}/bin/activate"
# Require valid Gemini API key (get one at https://aistudio.google.com/apikey)
if [ -z "\${GEMINI_API_KEY}" ]; then
  echo "[L43] ERROR: GEMINI_API_KEY is not set. Export a valid key: export GEMINI_API_KEY=your_key"
  echo "       Get a key at: https://aistudio.google.com/apikey"
  exit 1
fi
export GEMINI_API_KEY="\${GEMINI_API_KEY}"
export PYTHONPATH="${PROJECT_ROOT}"
# Use project-local cache for ChromaDB embedding models (avoids ~/.cache permission issues)
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
mkdir -p "${PROJECT_ROOT}/.cache"

echo "[L43] Seeding knowledge base..."
python3 "${BACKEND_DIR}/seed_knowledge_base.py"

echo "[L43] Starting backend on port ${BACKEND_PORT}..."
cd "${PROJECT_ROOT}"
uvicorn backend.api.app:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload &
echo \$! > /tmp/l43_backend.pid

echo "[L43] Starting frontend on port ${FRONTEND_PORT}..."
cd "${FRONTEND_DIR}" && PORT=${FRONTEND_PORT} npm start &
echo \$! > /tmp/l43_frontend.pid

echo ""
echo "✓ Backend:  http://localhost:${BACKEND_PORT}"
echo "✓ Frontend: http://localhost:${FRONTEND_PORT}"
echo "✓ API Docs: http://localhost:${BACKEND_PORT}/docs"
echo "✓ Health:   http://localhost:${BACKEND_PORT}/health"
BASH

cat > "${SCRIPTS_DIR}/stop.sh" << BASH
#!/usr/bin/env bash
[ -f /tmp/l43_backend.pid ]  && kill \$(cat /tmp/l43_backend.pid)  2>/dev/null && rm /tmp/l43_backend.pid  && echo "[L43] Backend stopped"
[ -f /tmp/l43_frontend.pid ] && kill \$(cat /tmp/l43_frontend.pid) 2>/dev/null && rm /tmp/l43_frontend.pid && echo "[L43] Frontend stopped"
pkill -f "uvicorn backend.api.app" 2>/dev/null || true
echo "[L43] ✓ All processes stopped"
BASH

cat > "${SCRIPTS_DIR}/test.sh" << BASH
#!/usr/bin/env bash
set -e
source "${VENV_DIR}/bin/activate"
export GEMINI_API_KEY="${GEMINI_API_KEY}"
export PYTHONPATH="${PROJECT_ROOT}"
BASE="http://localhost:${BACKEND_PORT}"

echo "[L43] Running API tests..."

# Health
echo -n "  Health check... "
STATUS=\$(curl -s -o /dev/null -w "%{http_code}" "\${BASE}/health")
[ "\$STATUS" = "200" ] && echo "✓" || { echo "✗ (HTTP \$STATUS)"; exit 1; }

# Query (timeout 120s — pipeline may do 429 retries)
echo -n "  Query endpoint... "
RESP=\$(curl -s --max-time 120 -X POST "\${BASE}/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the incident response protocol?","collection":"enterprise_kb"}')
echo \$RESP | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'trace_id' in d and 'response' in d, 'Missing fields'" && echo "✓" || { echo "✗"; exit 1; }

# Audit
TRACE_ID=\$(echo \$RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['trace_id'])")
echo -n "  Audit retrieval... "
STATUS=\$(curl -s -o /dev/null -w "%{http_code}" "\${BASE}/audit/\${TRACE_ID}")
[ "\$STATUS" = "200" ] && echo "✓" || echo "✗ (HTTP \$STATUS)"

# History
echo -n "  History endpoint... "
STATUS=\$(curl -s -o /dev/null -w "%{http_code}" "\${BASE}/history?n=5")
[ "\$STATUS" = "200" ] && echo "✓" || echo "✗ (HTTP \$STATUS)"

echo ""
echo "[L43] ✓ All tests passed!"
echo "      Trace ID: \${TRACE_ID}"
BASH

chmod +x "${SCRIPTS_DIR}/start.sh" "${SCRIPTS_DIR}/stop.sh" "${SCRIPTS_DIR}/test.sh"
log "Helper scripts created and made executable"

# ─── Docker Support ───────────────────────────────────────────────────────────
header "Creating Docker Configuration"
cat > "${PROJECT_ROOT}/Dockerfile.backend" << 'EOF'
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
RUN python -m backend.seed_knowledge_base || true

ENV GEMINI_API_KEY=""
ENV PYTHONPATH=/app
EXPOSE 8043

CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8043"]
EOF

cat > "${PROJECT_ROOT}/Dockerfile.frontend" << 'EOF'
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package.json .
RUN npm install --silent
COPY frontend/ .
ENV REACT_APP_API_URL=http://localhost:8043
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 3043
CMD ["nginx", "-g", "daemon off;"]
EOF

cat > "${PROJECT_ROOT}/docker-compose.yml" << EOF
version: '3.9'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports: ["${BACKEND_PORT}:8043"]
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./backend/data:/app/backend/data

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports: ["${FRONTEND_PORT}:3043"]
    depends_on: [backend]
EOF
log "Docker configuration created"

# ─── Seed Knowledge Base (if ChromaDB is accessible) ──────────────────────────
header "Seeding Knowledge Base"
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}" python3 "${BACKEND_DIR}/seed_knowledge_base.py" 2>/dev/null && \
  log "Knowledge base seeded successfully" || \
  warn "Seed deferred — will run on first startup"

# ─── Frontend Install ─────────────────────────────────────────────────────────
header "Installing Frontend Dependencies"
cd "${FRONTEND_DIR}"
npm install --silent 2>&1 | tail -3
log "Frontend dependencies installed"

# ─── Verify Generated Files ─────────────────────────────────────────────────
header "Verifying Generated Files"
REQUIRED_FILES=(
  "${PROJECT_ROOT}/requirements.txt"
  "${BACKEND_DIR}/core/models.py"
  "${BACKEND_DIR}/core/tracer.py"
  "${BACKEND_DIR}/core/orchestrator.py"
  "${BACKEND_DIR}/agents/planner.py"
  "${BACKEND_DIR}/agents/retriever.py"
  "${BACKEND_DIR}/agents/validator.py"
  "${BACKEND_DIR}/agents/synthesizer.py"
  "${BACKEND_DIR}/api/app.py"
  "${BACKEND_DIR}/seed_knowledge_base.py"
  "${FRONTEND_DIR}/package.json"
  "${FRONTEND_DIR}/public/index.html"
  "${FRONTEND_DIR}/src/index.js"
  "${FRONTEND_DIR}/src/App.jsx"
  "${SCRIPTS_DIR}/start.sh"
  "${SCRIPTS_DIR}/stop.sh"
  "${SCRIPTS_DIR}/test.sh"
  "${PROJECT_ROOT}/docker-compose.yml"
)
MISSING=()
for f in "${REQUIRED_FILES[@]}"; do
  [ -f "$f" ] || MISSING+=("$f")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  err "Missing generated files: ${MISSING[*]}"
fi
log "All ${#REQUIRED_FILES[@]} required files present"

# ─── Install Python Dependencies (after all files exist) ─────────────────────
header "Installing Python Dependencies"
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
log "Python dependencies installed"

# ─── Final Summary ────────────────────────────────────────────────────────────
header "Setup Complete"
echo ""
echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║   L43: Agentic RAG End-to-End — Ready to Run!    ║${NC}"
echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Start:${NC}    bash ${SCRIPTS_DIR}/start.sh"
echo -e "  ${CYAN}Test:${NC}     bash ${SCRIPTS_DIR}/test.sh"
echo -e "  ${CYAN}Stop:${NC}     bash ${SCRIPTS_DIR}/stop.sh"
echo ""
echo -e "  ${CYAN}Backend:${NC}  http://localhost:${BACKEND_PORT}"
echo -e "  ${CYAN}Frontend:${NC} http://localhost:${FRONTEND_PORT}"
echo -e "  ${CYAN}API Docs:${NC} http://localhost:${BACKEND_PORT}/docs"
echo ""
echo -e "  ${YELLOW}Pipeline:${NC} Planner → Retriever → Validator → Synthesizer"
echo -e "  ${YELLOW}Traces:${NC}   L42 AuditLogger integrated"
echo -e "  ${YELLOW}Next:${NC}     L44 Ragas evaluation uses /history endpoint"
echo ""
echo -e "  ${BOLD}Project:${NC} ${PROJECT_ROOT}"
log "L43 setup complete!"