import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import os
import asyncio
from functools import lru_cache
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class SummarizerService:
    """Multi-strategy text summarization for context compression"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        logger.info("SummarizerService initialized")
    
    async def extractive_summarize(
        self, 
        text: str, 
        ratio: float = 0.3,
        min_sentences: int = 2
    ) -> str:
        """
        Extractive summarization: select most important sentences
        Fast, preserves exact wording
        """
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= min_sentences:
                return text
            
            # Calculate sentence scores using TF-IDF
            scores = self._score_sentences(sentences)
            
            # Select top sentences
            target_count = max(min_sentences, int(len(sentences) * ratio))
            top_indices = np.argsort(scores)[-target_count:]
            
            # Maintain original order
            selected_sentences = [sentences[i] for i in sorted(top_indices)]
            
            summary = ' '.join(selected_sentences)
            logger.info(f"Extractive: {len(sentences)} → {len(selected_sentences)} sentences")
            
            return summary
            
        except Exception as e:
            logger.error(f"Extractive summarization error: {e}")
            return text[:len(text)//2]  # Fallback: truncate
    
    def _score_sentences(self, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF and position"""
        if len(sentences) == 0:
            return np.array([])
        
        # TF-IDF scoring
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except:
            # Fallback: simple word count
            sentence_scores = np.array([len(s.split()) for s in sentences])
        
        # Position bonus (first and last sentences often important)
        position_weights = np.ones(len(sentences))
        position_weights[0] = 1.5  # First sentence bonus
        position_weights[-1] = 1.3  # Last sentence bonus
        
        return sentence_scores * position_weights
    
    async def abstractive_summarize(
        self, 
        text: str, 
        target_ratio: float = 0.5,
        timeout: float = 10.0
    ) -> str:
        """
        Abstractive summarization using Gemini
        Slower, more compression, may lose nuance
        """
        try:
            # Calculate target length
            current_length = len(text)
            target_length = int(current_length * target_ratio)
            
            prompt = f"""Summarize the following text concisely, reducing it to approximately {target_length} characters while preserving all key information and technical details.

Text to summarize:
{text}

Concise summary:"""
            
            # Generate with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config={'temperature': 0.3, 'max_output_tokens': 2048}
                ),
                timeout=timeout
            )
            
            summary = response.text.strip()
            
            logger.info(f"Abstractive: {current_length} → {len(summary)} chars")
            return summary
            
        except asyncio.TimeoutError:
            logger.warning("Abstractive summarization timeout, using extractive fallback")
            return await self.extractive_summarize(text, ratio=target_ratio)
        except Exception as e:
            logger.error(f"Abstractive summarization error: {e}")
            return await self.extractive_summarize(text, ratio=target_ratio)
    
    async def hybrid_summarize(
        self, 
        text: str, 
        ratio: float = 0.4
    ) -> str:
        """
        Hybrid: extractive selection then abstractive compression
        Balanced speed and quality
        """
        try:
            # Step 1: Extractive selection (get to 60% of target)
            extractive_ratio = ratio * 1.5
            extracted = await self.extractive_summarize(text, ratio=extractive_ratio)
            
            # Step 2: Abstractive compression of extracted content
            if len(extracted) < len(text) * 0.8:  # Only if extraction helped
                summary = await self.abstractive_summarize(extracted, target_ratio=0.8)
                logger.info(f"Hybrid: {len(text)} → {len(summary)} chars")
                return summary
            else:
                return extracted
                
        except Exception as e:
            logger.error(f"Hybrid summarization error: {e}")
            return await self.extractive_summarize(text, ratio=ratio)
    
    async def summarize(
        self, 
        text: str, 
        strategy: str = "extractive",
        target_ratio: float = 0.3
    ) -> Dict:
        """
        Main summarization interface
        Returns summary with metadata
        """
        import time
        start_time = time.time()
        
        original_length = len(text)
        
        # Select strategy
        if strategy == "extractive":
            summary = await self.extractive_summarize(text, ratio=target_ratio)
        elif strategy == "abstractive":
            summary = await self.abstractive_summarize(text, target_ratio=target_ratio)
        elif strategy == "hybrid":
            summary = await self.hybrid_summarize(text, ratio=target_ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        compression_time = time.time() - start_time
        compressed_length = len(summary)
        compression_ratio = original_length / compressed_length if compressed_length > 0 else 1.0
        
        return {
            "original_text": text,
            "summary": summary,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": round(compression_ratio, 2),
            "strategy": strategy,
            "target_ratio": target_ratio,
            "compression_time_ms": round(compression_time * 1000, 2)
        }

# Global instance
summarizer = SummarizerService()
