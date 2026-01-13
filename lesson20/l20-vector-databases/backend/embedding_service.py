import google.generativeai as genai
from typing import List
import asyncio
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = 'models/text-embedding-004'
        logger.info("Gemini Embedding Service initialized")
    
    async def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Embed texts - Gemini API requires individual calls, so we do them concurrently"""
        
        async def embed_single(text: str) -> List[float]:
            """Embed a single text"""
            try:
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.model,
                    content=text,
                    task_type=task_type
                )
                
                # Extract embedding from response
                if isinstance(result, dict):
                    if 'embedding' in result:
                        emb = result['embedding']
                        # Handle different formats
                        if isinstance(emb, dict) and 'values' in emb:
                            return emb['values']
                        elif isinstance(emb, list):
                            # Ensure it's flat
                            if len(emb) > 0 and isinstance(emb[0], list):
                                return emb[0]
                            return emb
                        else:
                            return emb
                    elif 'values' in result:
                        return result['values']
                
                raise RuntimeError(f"Unexpected embedding response format: {type(result)}")
            except Exception as e:
                logger.error(f"Error embedding text '{text[:50]}...': {e}")
                raise RuntimeError(f"Failed to generate embedding: {str(e)}")
        
        # Process all texts concurrently with a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        async def embed_with_limit(text: str) -> List[float]:
            async with semaphore:
                return await embed_single(text)
        
        # Create tasks for all texts
        tasks = [embed_with_limit(text) for text in texts]
        
        # Wait for all embeddings to complete, return exceptions instead of raising
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        embeddings = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to embed text {idx}: {result}")
                raise RuntimeError(f"Failed to generate embedding for text {idx}: {str(result)}")
            elif not isinstance(result, list):
                logger.error(f"Unexpected result type for text {idx}: {type(result)}")
                raise RuntimeError(f"Embedding {idx} is not a list: {type(result)}")
            elif len(result) == 0:
                logger.error(f"Empty embedding for text {idx}")
                raise RuntimeError(f"Embedding {idx} is empty")
            elif not isinstance(result[0], (float, int)):
                logger.error(f"Non-numeric embedding for text {idx}: {type(result[0])}")
                raise RuntimeError(f"Embedding {idx} contains non-numeric values: {type(result[0])}")
            else:
                embeddings.append(result)
        
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
        
        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            # Handle response: result['embedding'] is a dict with 'values' key
            if isinstance(result, dict) and 'embedding' in result:
                if 'values' in result['embedding']:
                    return result['embedding']['values']
                else:
                    return result['embedding']
            if isinstance(result, dict) and 'values' in result:
                return result['values']
            raise RuntimeError(f"Unexpected embedding response format: {result.keys() if isinstance(result, dict) else type(result)}")
        except Exception as e:
            logger.error(f"Query embedding error: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}")
