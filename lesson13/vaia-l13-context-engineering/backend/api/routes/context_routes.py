from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from utils.token_counter import token_counter
from services.summarizer import summarizer
from services.context_optimizer import context_optimizer
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class TokenCountRequest(BaseModel):
    text: str = Field(..., description="Text to count tokens for")
    max_tokens: Optional[int] = Field(30000, description="Maximum token limit")

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    strategy: str = Field("extractive", description="Summarization strategy: extractive, abstractive, hybrid")
    target_ratio: float = Field(0.3, description="Target compression ratio (0.1-0.9)")

class OptimizeContextRequest(BaseModel):
    text: str = Field(..., description="Context to optimize")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens (uses default if not provided)")
    preserve_quality: bool = Field(True, description="Attempt to preserve quality during compression")

# Routes
@router.post("/count-tokens")
async def count_tokens(request: TokenCountRequest):
    """Count tokens in provided text"""
    try:
        analysis = token_counter.analyze(request.text, request.max_tokens)
        return {
            "success": True,
            "data": analysis
        }
    except Exception as e:
        logger.error(f"Token counting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """Summarize text using specified strategy"""
    try:
        if request.strategy not in ["extractive", "abstractive", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail="Strategy must be one of: extractive, abstractive, hybrid"
            )
        
        if not 0.1 <= request.target_ratio <= 0.9:
            raise HTTPException(
                status_code=400,
                detail="target_ratio must be between 0.1 and 0.9"
            )
        
        result = await summarizer.summarize(
            request.text,
            strategy=request.strategy,
            target_ratio=request.target_ratio
        )
        
        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-context")
async def optimize_context(request: OptimizeContextRequest):
    """Optimize context with intelligent compression"""
    try:
        result = await context_optimizer.optimize(
            request.text,
            max_tokens=request.max_tokens,
            preserve_quality=request.preserve_quality
        )
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Context optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "success": True,
        "data": {
            "service": "context-engineering",
            "version": "1.0.0",
            "strategies_available": ["extractive", "abstractive", "hybrid"],
            "default_max_tokens": 30000
        }
    }
