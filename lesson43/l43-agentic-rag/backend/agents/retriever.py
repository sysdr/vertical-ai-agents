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
