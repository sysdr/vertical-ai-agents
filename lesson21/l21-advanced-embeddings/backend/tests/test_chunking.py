import pytest
from app.services.chunking_service import ChunkingService

@pytest.mark.asyncio
async def test_semantic_chunking():
    service = ChunkingService()
    text = "This is sentence one. This is sentence two. This is a completely different topic. Another different sentence."
    
    chunks = await service.chunk_text(
        text=text,
        strategy="semantic",
        params={"threshold": 0.5}
    )
    
    assert len(chunks) >= 1
    assert all('text' in c for c in chunks)

@pytest.mark.asyncio
async def test_recursive_chunking():
    service = ChunkingService()
    text = "Short text that shouldn't be split."
    
    chunks = await service.chunk_text(
        text=text,
        strategy="recursive",
        params={"max_size": 1000}
    )
    
    assert len(chunks) == 1
