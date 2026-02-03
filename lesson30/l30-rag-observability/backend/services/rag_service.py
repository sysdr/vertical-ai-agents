import time
import asyncio
from typing import List, Dict
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from services.metrics_collector import metrics
from services.structured_logger import logger

class InstrumentedRAGService:
    """RAG pipeline with comprehensive observability"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        
        # Create sample collection
        self.collection = self.chroma_client.get_or_create_collection("knowledge_base")
        self._seed_knowledge_base()
    
    def _seed_knowledge_base(self):
        """Populate with sample documents"""
        docs = [
            "RAG combines retrieval and generation for contextual AI responses",
            "Observability requires logs, metrics, and distributed tracing",
            "Vector databases enable semantic search through embedding similarity",
            "LLM token costs scale with context size and generation length",
            "Production AI systems need monitoring for latency and accuracy",
        ]
        
        if self.collection.count() == 0:
            embeddings = self.embedder.encode(docs).tolist()
            self.collection.add(
                documents=docs,
                embeddings=embeddings,
                ids=[f"doc_{i}" for i in range(len(docs))]
            )
    
    async def process_query(self, query: str, trace_id: str) -> Dict:
        """Execute instrumented RAG pipeline"""
        pipeline_start = time.perf_counter()
        
        try:
            # Stage 1: Embedding
            embedding_result = await self._embed_query(query, trace_id)
            query_embedding = embedding_result["embedding"]
            
            # Stage 2: Retrieval
            retrieval_result = await self._retrieve_chunks(query_embedding, trace_id)
            chunks = retrieval_result["chunks"]
            
            # Stage 3: Context Augmentation
            augmentation_result = await self._augment_context(chunks, trace_id)
            context = augmentation_result["context"]
            
            # Stage 4: Generation
            generation_result = await self._generate_response(query, context, trace_id)
            
            # Record total pipeline latency
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            metrics.record("pipeline_latency_ms", total_latency, trace_id=trace_id)
            
            logger.log_stage(
                trace_id=trace_id,
                stage="completed",
                total_latency_ms=total_latency
            )
            
            return {
                "response": generation_result["response"],
                "metrics": {
                    "embedding_latency_ms": embedding_result["latency_ms"],
                    "retrieval_latency_ms": retrieval_result["latency_ms"],
                    "generation_latency_ms": generation_result["latency_ms"],
                    "total_latency_ms": total_latency,
                    "chunks_retrieved": len(chunks),
                    "avg_similarity": retrieval_result["avg_similarity"],
                    "tokens_used": generation_result["tokens"],
                    "cost_usd": generation_result["cost_usd"]
                },
                "trace_id": trace_id
            }
            
        except Exception as e:
            logger.log_error(trace_id=trace_id, stage="pipeline", error=e)
            metrics.increment("pipeline_errors", {"error_type": type(e).__name__})
            raise
    
    async def _embed_query(self, query: str, trace_id: str) -> Dict:
        """Instrumented embedding generation"""
        start = time.perf_counter()
        
        logger.log_stage(trace_id=trace_id, stage="embedding", query_length=len(query))
        
        # Simulate async execution
        await asyncio.sleep(0)
        embedding = self.embedder.encode(query).tolist()
        
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record("embedding_latency_ms", latency_ms, trace_id=trace_id)
        
        return {"embedding": embedding, "latency_ms": latency_ms}
    
    async def _retrieve_chunks(self, embedding: List[float], trace_id: str) -> Dict:
        """Instrumented vector retrieval"""
        start = time.perf_counter()
        
        logger.log_stage(trace_id=trace_id, stage="retrieving")
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=3
        )
        
        chunks = results['documents'][0]
        distances = results['distances'][0]
        similarities = [1 - d for d in distances]
        
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record("retrieval_latency_ms", latency_ms, trace_id=trace_id)
        metrics.record("chunks_retrieved", len(chunks), trace_id=trace_id)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        metrics.record("avg_similarity_score", avg_similarity, trace_id=trace_id)
        
        return {
            "chunks": chunks,
            "similarities": similarities,
            "avg_similarity": avg_similarity,
            "latency_ms": latency_ms
        }
    
    async def _augment_context(self, chunks: List[str], trace_id: str) -> Dict:
        """Instrumented context building"""
        start = time.perf_counter()
        
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        context_tokens = len(context.split())
        
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record("augmentation_latency_ms", latency_ms, trace_id=trace_id)
        metrics.record("context_tokens", context_tokens, trace_id=trace_id)
        
        logger.log_stage(
            trace_id=trace_id,
            stage="augmenting",
            context_tokens=context_tokens
        )
        
        return {"context": context, "tokens": context_tokens, "latency_ms": latency_ms}
    
    async def _generate_response(self, query: str, context: str, trace_id: str) -> Dict:
        """Instrumented LLM generation"""
        start = time.perf_counter()
        
        logger.log_stage(trace_id=trace_id, stage="generating")
        
        prompt = f"""Using the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        
        prompt_tokens = len(prompt.split())
        completion_tokens = 0
        total_tokens = prompt_tokens
        cost_usd = 0.0
        response_text = ""
        error: Exception | None = None
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            response_text = response.text
            completion_tokens = len(response_text.split())
            total_tokens = prompt_tokens + completion_tokens
            
            # Gemini Pro pricing estimate
            cost_usd = (prompt_tokens * 0.000125 + completion_tokens * 0.000375) / 1000
        except Exception as exc:  # noqa: BLE001 - best effort fallback for demo
            error = exc
            fallback = self._fallback_response(query, context)
            response_text = fallback["text"]
            completion_tokens = fallback["completion_tokens"]
            total_tokens = prompt_tokens + completion_tokens
            # cost remains 0.0
            
            logger.log_error(
                trace_id=trace_id,
                stage="generating",
                error=exc,
                message="Falling back to offline response"
            )
            metrics.increment("llm_fallbacks", {"error_type": type(exc).__name__})
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        metrics.record("generation_latency_ms", latency_ms, trace_id=trace_id)
        metrics.record("prompt_tokens", prompt_tokens, trace_id=trace_id)
        metrics.record("completion_tokens", completion_tokens, trace_id=trace_id)
        metrics.record("llm_cost_usd", cost_usd, trace_id=trace_id)
        
        logger.log_metric(
            trace_id=trace_id,
            metric_name="llm_usage",
            value=total_tokens,
            cost_usd=cost_usd,
            had_error=bool(error)
        )
        
        return {
            "response": response_text,
            "tokens": total_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms
        }
    
    def _fallback_response(self, query: str, context: str) -> Dict[str, float | str]:
        """Produce a deterministic response when the external LLM fails."""
        summary_lines = [
            "⚠️ Live LLM response unavailable — using local observability knowledge.",
            "",
            "Key Observations:",
        ]
        if context:
            summary_lines.append(context)
        summary_lines.extend(
            [
                "",
                "Reminder: configure a valid GEMINI_API_KEY to enable real-time generation.",
            ]
        )
        text = "\n".join(summary_lines)
        return {
            "text": text,
            "completion_tokens": len(text.split()),
        }
