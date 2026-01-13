from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
import json
import os

from vector_store import VectorStore
from embedding_service import EmbeddingService
from indexing_pipeline import IndexingPipeline
from search_engine import SearchEngine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L20: Vector Database API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Embedding functionality will not work.")

# Initialize components
vector_store = VectorStore(persist_directory="./data/chromadb")
embedding_service = EmbeddingService(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
if embedding_service:
    indexing_pipeline = IndexingPipeline(vector_store, embedding_service)
    search_engine = SearchEngine(vector_store, embedding_service)
else:
    indexing_pipeline = None
    search_engine = None

# WebSocket connections
active_connections: List[WebSocket] = []

class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    n_results: int = 10
    filters: Optional[Dict] = None

class IndexRequest(BaseModel):
    text: str
    source: str
    collection: str = "default"
    metadata: Optional[Dict] = None

@app.post("/index")
async def index_document(request: IndexRequest):
    """Index a document into vector store"""
    if not indexing_pipeline:
        raise HTTPException(status_code=503, detail="Embedding service not configured. Please set GEMINI_API_KEY environment variable.")
    try:
        result = await indexing_pipeline.index_document(
            document_text=request.text,
            source=request.source,
            collection_name=request.collection,
            metadata=request.metadata
        )
        await broadcast_stats()
        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Indexing error: {error_msg}")
        # Check for API key errors - more comprehensive detection
        error_lower = error_msg.lower()
        if any(keyword in error_lower for keyword in ["api key", "api_key", "expired", "invalid", "api_key_invalid"]):
            raise HTTPException(
                status_code=503, 
                detail="API key expired or invalid. Please set a valid GEMINI_API_KEY environment variable. Get your key from: https://makersuite.google.com/app/apikey"
            )
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/search")
async def search(request: SearchRequest):
    """Semantic search"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Embedding service not configured. Please set GEMINI_API_KEY environment variable.")
    try:
        results = await search_engine.search(
            query=request.query,
            collection_name=request.collection,
            n_results=request.n_results,
            filters=request.filters
        )
        return {"results": results, "query": request.query}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        # Check for API key errors - more comprehensive detection
        error_lower = error_msg.lower()
        if any(keyword in error_lower for keyword in ["api key", "api_key", "expired", "invalid", "api_key_invalid"]):
            raise HTTPException(
                status_code=503, 
                detail="API key expired or invalid. Please set a valid GEMINI_API_KEY environment variable. Get your key from: https://makersuite.google.com/app/apikey"
            )
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/stats/{collection}")
async def get_stats(collection: str):
    """Get collection statistics"""
    try:
        stats = vector_store.get_stats(collection)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_default_stats():
    """Get default collection statistics"""
    try:
        stats = vector_store.get_stats("default")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections"""
    collections = vector_store.list_collections()
    return {"collections": collections}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection (useful for re-indexing with correct embeddings)"""
    try:
        vector_store.delete_collection(collection_name)
        await broadcast_stats()
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")
        
        # Send initial stats immediately
        try:
            stats = vector_store.get_stats("default")
            await websocket.send_json(stats)
        except Exception as e:
            logger.debug(f"Error sending initial stats: {e}")
        
        # Keep connection alive
        while True:
            try:
                # Use receive with timeout to prevent hanging
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong
                if data == "ping":
                    try:
                        await websocket.send_text("pong")
                    except:
                        break  # Connection closed
            except asyncio.TimeoutError:
                # Send a keepalive stats update
                try:
                    stats = vector_store.get_stats("default")
                    await websocket.send_json(stats)
                except:
                    break  # Connection closed
            except Exception as e:
                # Connection closed or error - this is normal when client disconnects
                break
    except Exception as e:
        # Only log unexpected errors, not normal disconnections
        if "1000" not in str(e) and "1001" not in str(e):
            logger.debug(f"WebSocket error: {type(e).__name__}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        try:
            await websocket.close()
        except:
            pass
        logger.debug(f"WebSocket disconnected. Total connections: {len(active_connections)}")

async def broadcast_stats():
    """Broadcast stats to all connected clients"""
    if active_connections:
        stats = vector_store.get_stats("default")
        for connection in active_connections:
            try:
                await connection.send_json(stats)
            except:
                pass

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vector-database"}
