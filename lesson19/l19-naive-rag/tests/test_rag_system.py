"""Test suite for Naïve RAG system"""
import pytest
from backend.knowledge_base import KnowledgeBase
from backend.document_processor import DocumentProcessor
from backend.query_engine import QueryEngine

def test_document_chunking():
    processor = DocumentProcessor(chunk_size=100, overlap=20)
    text = "A" * 250
    chunks = processor.chunk_text(text)
    
    # With chunk_size=100, overlap=20, step=80:
    # Chunk 1: 0-100, Chunk 2: 80-180, Chunk 3: 160-250, Chunk 4: 240-250
    assert len(chunks) == 4
    assert chunks[0]['size'] == 100
    assert chunks[0]['position'] == 0
    assert chunks[-1]['position'] == 240

def test_knowledge_base_indexing():
    kb = KnowledgeBase()
    chunks = [
        {'text': 'Python is a programming language', 'position': 0},
        {'text': 'RAG stands for Retrieval Augmented Generation', 'position': 50}
    ]
    
    chunk_ids = kb.add_chunks(chunks, 'doc1')
    assert len(chunk_ids) == 2
    assert kb.metrics['total_chunks'] == 2

def test_keyword_retrieval():
    kb = KnowledgeBase()
    chunks = [
        {'text': 'Machine learning is a subset of AI', 'position': 0},
        {'text': 'Deep learning uses neural networks', 'position': 50},
        {'text': 'Python is great for machine learning', 'position': 100}
    ]
    kb.add_chunks(chunks, 'doc1')
    
    results = kb.retrieve('machine learning', top_k=2)
    assert len(results) <= 2
    assert results[0]['score'] > 0

def test_query_engine_pipeline():
    """Integration test - requires API key"""
    pass  # Skip in unit tests

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
