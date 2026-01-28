import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import GroundingValidator, CitationExtractor, AdversarialTester

def test_grounding_validator():
    """Test context relevance detection"""
    validator = GroundingValidator()
    
    # Relevant context
    result = validator.is_relevant(
        "What is RAG?",
        "RAG systems combine retrieval with generation"
    )
    assert result["is_relevant"] == True
    assert result["relevance_score"] > 0.3
    
    # Irrelevant context
    result = validator.is_relevant(
        "What is Python?",
        "The history of ancient Rome spans centuries"
    )
    assert result["is_relevant"] == False

def test_citation_extractor():
    """Test citation extraction and validation"""
    extractor = CitationExtractor()
    
    # Valid citations
    answer = "RAG is useful [Source 1]. It combines retrieval and generation [Source 2]."
    citations = extractor.extract_citations(answer)
    assert citations == [1, 2]
    
    # Validation with correct sources
    validation = extractor.validate_citations(answer, num_sources=3)
    assert validation["valid"] == True
    assert validation["citations_found"] == [1, 2]
    
    # Validation with invalid sources
    validation = extractor.validate_citations(answer, num_sources=1)
    assert validation["valid"] == False
    assert 2 in validation["invalid_citations"]

def test_adversarial_tester():
    """Test misleading context generation"""
    tester = AdversarialTester()
    
    # Python question gets JavaScript context
    context = tester.generate_misleading_context("How does Python async work?")
    assert "JavaScript" in context or "promises" in context
    
    # Default fallback
    context = tester.generate_misleading_context("Random question")
    assert len(context) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
