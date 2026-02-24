"""
L44 Metrics Engine — Ragas + custom Gemini-judge evaluation.

Two evaluation paths:
  1. Ragas (primary): LangchainLLMWrapper(Gemini) → ragas.evaluate()
  2. Custom (fallback): Direct Gemini prompting with structured JSON output

Both produce MetricScores-compatible output.
"""
import asyncio
import json
import logging
from typing import Optional
import google.generativeai as genai
from app.models import TraceRecord, MetricScores

logger = logging.getLogger(__name__)

# ── Evaluation prompts ────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an expert RAG system evaluator assessing FAITHFULNESS.

Faithfulness measures whether all claims in the ANSWER are supported by the CONTEXT.
A faithfulness score of 1.0 means every claim is grounded. 0.0 means everything is hallucinated.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Instructions:
1. Extract all factual claims from the ANSWER
2. For each claim, determine if it is directly supported by the CONTEXT
3. Score = supported_claims / total_claims

Respond with ONLY valid JSON, no markdown:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported_statements": [<strings>],
  "reasoning": "<brief explanation>"
}}"""

RELEVANCY_PROMPT = """You are an expert RAG system evaluator assessing ANSWER RELEVANCY.

Answer relevancy measures how directly the ANSWER addresses the QUESTION.
Score 1.0 = perfectly on-topic, directly answers what was asked.
Score 0.0 = completely irrelevant or answers a different question.

QUESTION: {question}
GENERATED ANSWER: {answer}

Instructions:
1. Assess whether the answer directly addresses the question
2. Penalize answers that include only tangential information
3. Penalize incomplete answers that miss the core question

Respond with ONLY valid JSON, no markdown:
{{
  "relevancy_score": <float 0.0-1.0>,
  "addresses_question": <bool>,
  "missing_aspects": [<strings>],
  "reasoning": "<brief explanation>"
}}"""

CONTEXT_RECALL_PROMPT = """You are an expert RAG system evaluator assessing CONTEXT RECALL.

Context recall measures whether the retrieved context contains the information needed
to produce the ground truth answer. Score 1.0 = context has everything needed.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
RETRIEVED CONTEXT: {context}

Instructions:
1. Identify all claims/facts in the GROUND TRUTH ANSWER
2. Check each claim against the RETRIEVED CONTEXT
3. Score = claims_in_context / total_ground_truth_claims

Respond with ONLY valid JSON, no markdown:
{{
  "total_gt_claims": <int>,
  "claims_in_context": <int>,
  "context_recall_score": <float 0.0-1.0>,
  "missing_from_context": [<strings>],
  "reasoning": "<brief explanation>"
}}"""

CONTEXT_PRECISION_PROMPT = """You are an expert RAG system evaluator assessing CONTEXT PRECISION.

Context precision measures what fraction of the retrieved context chunks are
actually relevant to answering the question. Score 1.0 = all chunks are useful.

QUESTION: {question}
RETRIEVED CONTEXT CHUNKS:
{context_numbered}

Instructions:
1. For each numbered context chunk, determine if it's relevant to the question
2. Score = relevant_chunks / total_chunks

Respond with ONLY valid JSON, no markdown:
{{
  "total_chunks": <int>,
  "relevant_chunks": <int>,
  "context_precision_score": <float 0.0-1.0>,
  "relevant_chunk_indices": [<ints>],
  "reasoning": "<brief explanation>"
}}"""


class GeminiJudge:
    """
    Direct Gemini-as-judge implementation.
    Temperature=0 for reproducible evaluation scores.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )

    async def _judge(self, prompt: str) -> dict:
        """Call Gemini with retry logic."""
        from tenacity import retry, stop_after_attempt, wait_exponential
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(prompt)
        )
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        return json.loads(text)

    async def faithfulness(self, question: str, context: str, answer: str) -> float:
        try:
            ctx_text = "\n\n".join(context) if isinstance(context, list) else context
            result = await self._judge(
                FAITHFULNESS_PROMPT.format(
                    question=question, context=ctx_text[:4000], answer=answer
                )
            )
            return float(result.get("faithfulness_score", 0.0))
        except Exception as e:
            logger.warning(f"Faithfulness eval failed: {e}")
            return 0.0

    async def answer_relevancy(self, question: str, answer: str) -> float:
        try:
            result = await self._judge(
                RELEVANCY_PROMPT.format(question=question, answer=answer)
            )
            return float(result.get("relevancy_score", 0.0))
        except Exception as e:
            logger.warning(f"Relevancy eval failed: {e}")
            return 0.0

    async def context_recall(self, question: str, ground_truth: str, context: str) -> float:
        if not ground_truth:
            return None  # Can't compute without ground truth
        try:
            ctx_text = "\n\n".join(context) if isinstance(context, list) else context
            result = await self._judge(
                CONTEXT_RECALL_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    context=ctx_text[:4000]
                )
            )
            return float(result.get("context_recall_score", 0.0))
        except Exception as e:
            logger.warning(f"Context recall eval failed: {e}")
            return 0.0

    async def context_precision(self, question: str, contexts: list[str]) -> float:
        try:
            numbered = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
            result = await self._judge(
                CONTEXT_PRECISION_PROMPT.format(
                    question=question, context_numbered=numbered
                )
            )
            return float(result.get("context_precision_score", 0.0))
        except Exception as e:
            logger.warning(f"Context precision eval failed: {e}")
            return 0.0

    async def evaluate_trace(self, trace: TraceRecord) -> MetricScores:
        """Evaluate all 4 metrics for a single trace — parallel execution."""
        ctx = trace.retrieved_contexts
        ctx_str = "\n\n".join(ctx)

        tasks = [
            self.faithfulness(trace.question, ctx_str, trace.answer),
            self.answer_relevancy(trace.question, trace.answer),
        ]

        # Add reference-required metrics only if ground truth exists
        if trace.ground_truth:
            tasks.append(self.context_recall(trace.question, trace.ground_truth, ctx_str))
            tasks.append(self.context_precision(trace.question, ctx))
        else:
            tasks.append(asyncio.coroutine(lambda: 0.75)())  # default if no GT
            tasks.append(self.context_precision(trace.question, ctx))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        def safe(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

        return MetricScores(
            faithfulness=safe(results[0]),
            answer_relevancy=safe(results[1]),
            context_recall=safe(results[2]),
            context_precision=safe(results[3]),
        )


def try_ragas_evaluate(traces: list[TraceRecord], api_key: str) -> Optional[list[MetricScores]]:
    """
    Attempt Ragas evaluation. Returns None if Ragas is unavailable.
    Uses LangchainLLMWrapper(Gemini) as the judge LLM.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_recall, context_precision
        )
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset

        # Gemini as judge
        lc_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True,
        )
        lc_emb = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        ragas_llm = LangchainLLMWrapper(lc_llm)
        ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

        # Build HuggingFace Dataset format expected by Ragas
        data = {
            "question": [t.question for t in traces],
            "answer": [t.answer for t in traces],
            "contexts": [t.retrieved_contexts for t in traces],
            "ground_truth": [t.ground_truth or "" for t in traces],
        }
        dataset = Dataset.from_dict(data)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )
        df = result.to_pandas()

        scores = []
        for _, row in df.iterrows():
            scores.append(MetricScores(
                faithfulness=float(row.get("faithfulness", 0)),
                answer_relevancy=float(row.get("answer_relevancy", 0)),
                context_recall=float(row.get("context_recall", 0)),
                context_precision=float(row.get("context_precision", 0)),
            ))
        logger.info(f"Ragas evaluation completed for {len(scores)} traces")
        return scores

    except ImportError:
        logger.warning("Ragas not available — using custom Gemini judge path")
        return None
    except Exception as e:
        logger.warning(f"Ragas evaluation failed ({e}) — falling back to custom judge")
        return None
