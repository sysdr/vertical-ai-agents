import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from services.orchestrator import RAGOrchestrator
from models.pipeline import PipelineStage, StageResult

# Use placeholder for tests (synthesizer is mocked, no real API calls)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "test-placeholder")


def _mock_synthesize(documents, query, validation_scores, budget_available):
    """Mock synthesize that returns success without calling API"""
    context = " ".join(doc.get("content", "") for doc in documents)
    mock_response = f"Based on the documents: {context[:100]}... Answer to '{query}'."
    tokens = len(mock_response.split()) + 50
    return StageResult(
        success=True,
        data={
            "response": mock_response,
            "citations": [doc.get("source", "") for doc in documents],
            "confidence": validation_scores.get("overall_confidence", 0.9)
        },
        tokens_used=min(tokens, budget_available),
        metadata={"agent": "synthesizer", "sources_used": len(documents)}
    )


@pytest.fixture
def mock_synthesizer():
    """Patch SynthesizerAgent to use mock instead of real API"""
    with patch("services.orchestrator.SynthesizerAgent") as MockSynth:
        instance = MagicMock()
        instance.synthesize = AsyncMock(side_effect=_mock_synthesize)
        MockSynth.return_value = instance
        yield MockSynth


@pytest.mark.asyncio
async def test_simple_query(mock_synthesizer):
    """Test simple query processing"""
    orch = RAGOrchestrator(GEMINI_API_KEY, max_tokens=3000)
    result = await orch.process_query("What is RAG?")
    
    assert result.stage == PipelineStage.COMPLETED
    assert result.response is not None
    assert len(result.documents) > 0
    assert result.total_tokens <= 3000

@pytest.mark.asyncio
async def test_budget_enforcement(mock_synthesizer):
    """Test budget limits are respected"""
    orch = RAGOrchestrator(GEMINI_API_KEY, max_tokens=500)  # Very low budget
    result = await orch.process_query("Explain agentic RAG systems in detail")
    
    assert result.total_tokens <= 500

@pytest.mark.asyncio
async def test_validation_gate(mock_synthesizer):
    """Test validation stage works"""
    orch = RAGOrchestrator(GEMINI_API_KEY)
    result = await orch.process_query("What are vector embeddings?")
    
    assert result.validation_scores is not None
    assert "overall_confidence" in result.validation_scores
    assert result.validation_scores["overall_confidence"] >= 0.7

@pytest.mark.asyncio
async def test_all_agents_execute(mock_synthesizer):
    """Test all four agents execute successfully"""
    orch = RAGOrchestrator(GEMINI_API_KEY)
    result = await orch.process_query("Explain multi-agent systems")
    
    assert result.plan is not None  # Planner executed
    assert len(result.documents) > 0  # Retriever executed
    assert result.validation_scores is not None  # Validator executed
    assert result.response is not None  # Synthesizer executed

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
