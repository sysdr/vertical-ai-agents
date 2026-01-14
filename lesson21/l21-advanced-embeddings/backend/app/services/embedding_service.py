try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

import numpy as np
from typing import List, Dict
import os
import asyncio

class EmbeddingService:
    def __init__(self):
        # Initialize models lazily
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not initialize SentenceTransformer: {e}")
        
        # Configure Gemini
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    genai.configure(api_key=api_key)
            except Exception as e:
                print(f"Warning: Could not configure Gemini: {e}")
        
        self.available_models = {
            "sentence-transformer": {
                "name": "all-MiniLM-L6-v2",
                "dimension": 384,
                "speed": "fast",
                "use_case": "general purpose"
            },
            "gemini": {
                "name": "models/embedding-001",
                "dimension": 768,
                "speed": "medium",
                "use_case": "semantic understanding"
            },
            "lightweight": {
                "name": "paraphrase-MiniLM-L3-v2",
                "dimension": 384,
                "speed": "very fast",
                "use_case": "high throughput"
            }
        }
    
    async def embed_texts(self, texts: List[str], model: str = "sentence-transformer") -> List[List[float]]:
        """Generate embeddings for list of texts"""
        if model == "sentence-transformer":
            return await self._embed_sentence_transformer(texts)
        elif model == "gemini":
            return await self._embed_gemini(texts)
        elif model == "lightweight":
            return await self._embed_lightweight(texts)
        else:
            raise ValueError(f"Unknown model: {model}")
    
    async def _embed_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Use Sentence Transformers"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.sentence_model is None:
            raise ValueError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.sentence_model.encode(texts, batch_size=32)
        )
        return embeddings.tolist()
    
    async def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        """Use Gemini embeddings API"""
        if not GEMINI_AVAILABLE:
            raise ValueError("google-generativeai is not installed. Please install it with: pip install google-generativeai")
        embeddings = []
        
        # Batch processing for Gemini
        for text in texts:
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Gemini embedding error: {e}")
                # Fallback to zeros
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def _embed_lightweight(self, texts: List[str]) -> List[List[float]]:
        """Use lightweight model for speed"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError("sentence-transformers is not installed. Please install it with: pip install sentence-transformers")
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=64)
        )
        return embeddings.tolist()
    
    def get_model_metadata(self, model: str) -> Dict:
        """Get metadata about embedding model"""
        return self.available_models.get(model, {})
