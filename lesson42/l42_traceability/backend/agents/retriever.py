"""
RetrieverAgent: simulates semantic retrieval with trace instrumentation.
"""
import os, hashlib, time
from dataclasses import dataclass
import google.generativeai as genai
from backend.tracing.decorator import trace_step

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-flash"

# Simulated document corpus
CORPUS = [
    {"id": "doc_001", "text": "Neural networks learn by adjusting weights through backpropagation."},
    {"id": "doc_002", "text": "Transformer models use attention mechanisms to process sequences."},
    {"id": "doc_003", "text": "RAG systems retrieve documents before generating responses."},
    {"id": "doc_004", "text": "Enterprise AI requires compliance, auditability, and observability."},
    {"id": "doc_005", "text": "Vector databases store embeddings for semantic similarity search."},
    {"id": "doc_006", "text": "Agentic pipelines decompose complex queries into sub-tasks."},
]


@dataclass
class RetrievalResult:
    docs: list[dict]
    scores: list[float]
    strategy_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def to_trace_dict(self) -> dict:
        return {
            "output_summary": {
                "doc_ids": [d["id"] for d in self.docs],
                "top_score": max(self.scores, default=0.0),
                "strategy": self.strategy_used,
            },
            "decision": f"RETRIEVED:{len(self.docs)} STRATEGY:{self.strategy_used}",
            "confidence": max(self.scores, default=0.0),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_id": _MODEL,
        }


class RetrieverAgent:
    @trace_step("retriever", capture_input_fields=["query", "strategy"])
    async def retrieve(
        self, query: str, strategy: str = "DEFAULT", top_k: int = 3
    ) -> RetrievalResult:
        t0 = time.monotonic()

        # Simulate semantic scoring with keyword overlap
        scores = []
        query_words = set(query.lower().split())
        for doc in CORPUS:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / (len(query_words) + 1)
            scores.append(score)

        # Rerank: strict strategy boosts exact matches
        if strategy == "STRICT":
            scores = [s * 1.3 if s > 0 else 0 for s in scores]

        pairs = sorted(zip(scores, CORPUS), reverse=True)[:top_k]
        selected_scores = [p[0] for p in pairs]
        selected_docs = [p[1] for p in pairs]

        return RetrievalResult(
            docs=selected_docs,
            scores=selected_scores,
            strategy_used=strategy,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            input_tokens=len(query.split()),
            output_tokens=sum(len(d["text"].split()) for d in selected_docs),
        )
