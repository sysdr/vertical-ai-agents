from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os

from .services.embedding_service import EmbeddingService
from .services.chunking_service import ChunkingService
from .services.vector_service import VectorService
from .models.schemas import (
    ChunkRequest, ChunkResponse, EmbedRequest, EmbedResponse,
    CompareRequest, CompareResponse, DocumentUpload
)

app = FastAPI(title="L21 Advanced Embeddings & Chunking")

# CORS - Must be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Exception handler to ensure CORS headers on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# Initialize services
embedding_service = EmbeddingService()
chunking_service = ChunkingService()
vector_service = VectorService()

@app.on_event("startup")
async def startup_event():
    """Initialize NLTK data and services"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    await vector_service.initialize()
    print("✓ Services initialized")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "lesson": "L21",
        "services": {
            "embeddings": len(embedding_service.available_models),
            "chunking_strategies": len(chunking_service.strategies),
            "vector_collections": await vector_service.list_collections()
        }
    }

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_document(request: ChunkRequest):
    """Apply chunking strategy to document"""
    try:
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not request.strategy:
            raise HTTPException(status_code=400, detail="Strategy must be provided")
        
        chunks = await chunking_service.chunk_text(
            text=request.text,
            strategy=request.strategy,
            params=request.params or {}
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated. Check strategy and parameters.")
        
        return ChunkResponse(
            strategy=request.strategy,
            chunks=chunks,
            total_chunks=len(chunks),
            metadata=chunking_service.get_strategy_metadata(request.strategy)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chunking error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """Generate embeddings using specified model"""
    try:
        embeddings = await embedding_service.embed_texts(
            texts=request.texts,
            model=request.model
        )
        
        return EmbedResponse(
            model=request.model,
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            metadata=embedding_service.get_model_metadata(request.model)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/compare", response_model=CompareResponse)
async def compare_strategies(request: CompareRequest):
    """Compare different chunking and embedding strategies"""
    try:
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not request.strategies or len(request.strategies) == 0:
            raise HTTPException(status_code=400, detail="At least one strategy must be provided")
        if not request.models or len(request.models) == 0:
            raise HTTPException(status_code=400, detail="At least one model must be provided")
        
        results = {}
        
        for strategy in request.strategies:
            try:
                # Chunk the text
                chunks = await chunking_service.chunk_text(
                    text=request.text,
                    strategy=strategy,
                    params=request.params.get(strategy, {})
                )
                
                if not chunks or len(chunks) == 0:
                    results[strategy] = {"error": "No chunks generated"}
                    continue
                
                # Generate embeddings for all models
                strategy_results = {}
                for model in request.models:
                    try:
                        chunk_texts = [c["text"] for c in chunks]
                        embeddings = await embedding_service.embed_texts(chunk_texts, model)
                        
                        # Calculate quality metrics
                        metrics = await chunking_service.calculate_metrics(chunks, embeddings)
                        
                        strategy_results[model] = {
                            "chunks": chunks,
                            "embeddings_count": len(embeddings),
                            "metrics": metrics
                        }
                    except Exception as e:
                        print(f"Error processing model {model} for strategy {strategy}: {e}")
                        import traceback
                        traceback.print_exc()
                        strategy_results[model] = {
                            "chunks": chunks,
                            "embeddings_count": 0,
                            "metrics": {"error": str(e)}
                        }
                
                results[strategy] = strategy_results
            except Exception as e:
                print(f"Error processing strategy {strategy}: {e}")
                import traceback
                traceback.print_exc()
                results[strategy] = {"error": str(e)}
        
        return CompareResponse(results=results, total_comparisons=len(results))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    content = await file.read()
    text = content.decode('utf-8')
    
    # Process with default strategy
    chunks = await chunking_service.chunk_text(
        text=text,
        strategy="semantic",
        params={"threshold": 0.5}
    )
    
    # Store in vector database
    doc_id = await vector_service.store_document(
        document_id=file.filename,
        chunks=chunks,
        strategy="semantic"
    )
    
    return {
        "document_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(chunks)
    }

@app.get("/strategies")
async def list_strategies():
    """List available chunking strategies"""
    return {
        "strategies": list(chunking_service.strategies.keys()),
        "models": list(embedding_service.available_models.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
