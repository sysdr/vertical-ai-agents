import pytest
from models.document import Document
from services.validator_service import ValidatorService

@pytest.mark.asyncio
async def test_generate_pairs():
    validator = ValidatorService()
    docs = [
        Document(id="1", content="Test 1", score=0.9, source="A", metadata={}),
        Document(id="2", content="Test 2", score=0.8, source="B", metadata={}),
        Document(id="3", content="Test 3", score=0.7, source="C", metadata={})
    ]
    
    pairs = validator.generate_pairs(docs)
    assert len(pairs) == 3  # N(N-1)/2 = 3(2)/2 = 3

@pytest.mark.asyncio
async def test_compute_pair_hash():
    validator = ValidatorService()
    
    hash1 = validator.compute_pair_hash("doc1", "doc2")
    hash2 = validator.compute_pair_hash("doc2", "doc1")
    
    # Should be same regardless of order
    assert hash1 == hash2
