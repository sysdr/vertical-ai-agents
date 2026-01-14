import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
import re

class ChunkingService:
    def __init__(self):
        self.strategies = {
            "fixed": self._fixed_chunk,
            "semantic": self._semantic_chunk,
            "recursive": self._recursive_chunk,
            "sliding_window": self._sliding_window_chunk,
            "sentence": self._sentence_chunk
        }
    
    async def chunk_text(self, text: str, strategy: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply chunking strategy to text"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        chunker = self.strategies[strategy]
        chunks = await chunker(text, params)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk['index'] = i
            chunk['strategy'] = strategy
        
        return chunks
    
    async def _fixed_chunk(self, text: str, params: Dict) -> List[Dict]:
        """Fixed-size chunking (baseline)"""
        chunk_size = params.get('chunk_size', 500)
        overlap = params.get('overlap', 50)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "metadata": {"size": len(chunk_text)}
            })
            
            start += chunk_size - overlap
        
        return chunks
    
    async def _semantic_chunk(self, text: str, params: Dict) -> List[Dict]:
        """Semantic boundary chunking"""
        from sentence_transformers import SentenceTransformer
        
        threshold = params.get('threshold', 0.5)
        min_chunk_size = params.get('min_chunk_size', 100)
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            return [{"text": text, "start_pos": 0, "end_pos": len(text), "metadata": {}}]
        
        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        
        # Find semantic boundaries
        chunks = []
        current_chunk = [sentences[0]]
        current_start = 0
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [embeddings[i-1]], 
                [embeddings[i]]
            )[0][0]
            
            if similarity < threshold:
                # Boundary detected
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "start_pos": current_start,
                        "end_pos": current_start + len(chunk_text),
                        "metadata": {
                            "sentences": len(current_chunk),
                            "boundary_similarity": float(similarity)
                        }
                    })
                    current_start += len(chunk_text) + 1
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            else:
                current_chunk.append(sentences[i])
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_pos": current_start,
                "end_pos": current_start + len(chunk_text),
                "metadata": {"sentences": len(current_chunk)}
            })
        
        return chunks
    
    async def _recursive_chunk(self, text: str, params: Dict) -> List[Dict]:
        """Recursive hierarchical chunking"""
        max_size = params.get('max_size', 1000)
        overlap = params.get('overlap', 200)
        
        def _recursive_split(text: str, level: int = 0) -> List[Dict]:
            if len(text) <= max_size:
                return [{
                    "text": text,
                    "start_pos": 0,
                    "end_pos": len(text),
                    "metadata": {"level": level, "children": []}
                }]
            
            # Split at paragraph boundaries
            paragraphs = re.split(r'\n\n+', text)
            
            if len(paragraphs) == 1:
                # No paragraphs, split at sentence boundaries
                sentences = sent_tokenize(text)
                mid = len(sentences) // 2
                left = " ".join(sentences[:mid])
                right = " ".join(sentences[mid:])
                
                return _recursive_split(left, level + 1) + _recursive_split(right, level + 1)
            
            # Recursively process paragraphs
            chunks = []
            for para in paragraphs:
                if len(para) > max_size:
                    chunks.extend(_recursive_split(para, level + 1))
                else:
                    chunks.append({
                        "text": para,
                        "start_pos": 0,
                        "end_pos": len(para),
                        "metadata": {"level": level + 1, "children": []}
                    })
            
            return chunks
        
        return _recursive_split(text)
    
    async def _sliding_window_chunk(self, text: str, params: Dict) -> List[Dict]:
        """Sliding window with configurable stride"""
        window_size = params.get('window_size', 500)
        stride = params.get('stride', 250)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "start_pos": start,
                "end_pos": end,
                "metadata": {"window_size": window_size, "stride": stride}
            })
            
            start += stride
            
            if end == len(text):
                break
        
        return chunks
    
    async def _sentence_chunk(self, text: str, params: Dict) -> List[Dict]:
        """Sentence-level chunking"""
        sentences_per_chunk = params.get('sentences_per_chunk', 3)
        
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)
            
            chunks.append({
                "text": chunk_text,
                "start_pos": i,
                "end_pos": i + len(chunk_sentences),
                "metadata": {"sentences": len(chunk_sentences)}
            })
        
        return chunks
    
    async def calculate_metrics(self, chunks: List[Dict], embeddings: List[List[float]]) -> Dict:
        """Calculate quality metrics for chunks"""
        if not chunks or not embeddings:
            return {}
        
        chunk_sizes = [len(c['text']) for c in chunks]
        
        # Calculate coherence (average similarity between consecutive chunks)
        coherence = 0.0
        if len(embeddings) > 1:
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                similarities.append(sim)
            coherence = float(np.mean(similarities))
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": float(np.mean(chunk_sizes)),
            "std_chunk_size": float(np.std(chunk_sizes)),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "coherence_score": coherence
        }
    
    def get_strategy_metadata(self, strategy: str) -> Dict:
        """Get metadata about chunking strategy"""
        metadata = {
            "fixed": {
                "description": "Fixed-size chunks with overlap",
                "params": ["chunk_size", "overlap"],
                "use_case": "baseline comparison"
            },
            "semantic": {
                "description": "Boundary detection using embeddings",
                "params": ["threshold", "min_chunk_size"],
                "use_case": "preserving semantic meaning"
            },
            "recursive": {
                "description": "Hierarchical splitting",
                "params": ["max_size", "overlap"],
                "use_case": "structured documents"
            },
            "sliding_window": {
                "description": "Overlapping windows",
                "params": ["window_size", "stride"],
                "use_case": "dense coverage"
            },
            "sentence": {
                "description": "Sentence-level grouping",
                "params": ["sentences_per_chunk"],
                "use_case": "fine-grained retrieval"
            }
        }
        return metadata.get(strategy, {})
