from fastapi import APIRouter
import random

router = APIRouter()

@router.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for evaluation"""
    return {
        "queries": [
            "What is RAG?",
            "How does evaluation work?",
            "What metrics are important?",
            "Explain context precision",
            "What is faithfulness?",
            "How to measure relevance?",
            "What is RAGAS?",
            "Why evaluate RAG systems?",
            "What is multi-turn conversation?",
            "How does hybrid search work?"
        ]
    }

@router.get("/baseline")
async def get_baseline():
    """Get baseline metrics for comparison"""
    return {
        "baseline_name": "GPT-3.5 + ChromaDB",
        "faithfulness": 0.87,
        "answer_relevance": 0.82,
        "context_precision": 0.79,
        "context_recall": 0.75,
        "date": "2025-01-15"
    }
