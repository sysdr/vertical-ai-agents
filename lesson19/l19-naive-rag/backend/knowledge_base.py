"""In-memory knowledge base with keyword-based retrieval"""
import time
from typing import List, Dict, Optional
from collections import defaultdict
import re

class KnowledgeBase:
    def __init__(self):
        self.chunks: Dict[str, Dict] = {}
        self.chunk_counter = 0
        self.keyword_index: Dict[str, set] = defaultdict(set)
        self.metrics = {
            'total_chunks': 0,
            'total_queries': 0,
            'avg_retrieval_time_ms': 0
        }
    
    def add_chunks(self, chunks: List[Dict], doc_id: str) -> List[str]:
        """Add chunks to knowledge base with keyword indexing"""
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Store chunk with metadata
            self.chunks[chunk_id] = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'text': chunk['text'],
                'position': chunk['position'],
                'overlap': chunk.get('next_overlap', ''),
                'timestamp': time.time()
            }
            
            # Build keyword index
            keywords = self._extract_keywords(chunk['text'])
            for keyword in keywords:
                self.keyword_index[keyword].add(chunk_id)
            
            chunk_ids.append(chunk_id)
            self.chunk_counter += 1
        
        self.metrics['total_chunks'] = self.chunk_counter
        return chunk_ids
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks using keyword matching"""
        start_time = time.time()
        
        # Extract query keywords
        query_keywords = self._extract_keywords(query)
        
        # Score chunks based on keyword overlap
        chunk_scores = defaultdict(int)
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                for chunk_id in self.keyword_index[keyword]:
                    chunk_scores[chunk_id] += 1
        
        # Rank and retrieve top-k
        ranked_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = [
            {**self.chunks[chunk_id], 'score': score}
            for chunk_id, score in ranked_chunks
            if chunk_id in self.chunks
        ]
        
        # Update metrics
        retrieval_time = (time.time() - start_time) * 1000
        self.metrics['total_queries'] += 1
        self.metrics['avg_retrieval_time_ms'] = (
            (self.metrics['avg_retrieval_time_ms'] * (self.metrics['total_queries'] - 1) + retrieval_time)
            / self.metrics['total_queries']
        )
        
        return results
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text (simple tokenization)"""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split and filter stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        words = [w for w in text.split() if len(w) > 2 and w not in stopwords]
        return set(words)
    
    def get_stats(self) -> Dict:
        """Return knowledge base statistics"""
        return {
            'total_chunks': self.metrics['total_chunks'],
            'total_queries': self.metrics['total_queries'],
            'avg_retrieval_time_ms': round(self.metrics['avg_retrieval_time_ms'], 2),
            'unique_keywords': len(self.keyword_index),
            'chunks_per_document': self._get_chunks_per_doc()
        }
    
    def _get_chunks_per_doc(self) -> Dict[str, int]:
        """Count chunks per document"""
        doc_counts = defaultdict(int)
        for chunk in self.chunks.values():
            doc_counts[chunk['doc_id']] += 1
        return dict(doc_counts)
