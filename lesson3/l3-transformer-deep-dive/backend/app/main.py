"""
FastAPI backend for Transformer Visualizer.
Builds on L2 async patterns.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import List, Dict
import numpy as np
import time

from .config import settings
from .transformer import (
    AttentionConfig,
    TransformerEncoder,
    SimpleTokenizer
)

app = FastAPI(title="Transformer Deep Dive API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
tokenizer = SimpleTokenizer()
transformer_config = AttentionConfig(
    d_model=settings.d_model,
    num_heads=settings.num_heads,
    dropout=settings.dropout
)
transformer = TransformerEncoder(transformer_config, num_layers=settings.num_layers)

class TransformRequest(BaseModel):
    text: str
    visualize: bool = True

class TransformResponse(BaseModel):
    tokens: List[str]
    attention_weights: List[Dict]
    layer_outputs: List[List[List[float]]]
    latency_ms: float
    metadata: Dict

@app.get("/")
async def root():
    return {
        "service": "Transformer Deep Dive API",
        "version": "1.0.0",
        "lesson": "L3",
        "status": "operational"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/transform", response_model=TransformResponse)
async def transform_text(request: TransformRequest):
    """
    Process text through transformer and return attention weights.
    """
    start_time = time.time()
    
    # Tokenize
    token_ids = tokenizer.tokenize(request.text)
    tokens = [tokenizer.reverse_vocab.get(t, "<UNK>") for t in token_ids]
    
    # Create embeddings
    embeddings = tokenizer.create_embeddings(token_ids, settings.d_model)
    
    # Forward pass through transformer
    output, attention_weights = await transformer.forward(embeddings)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return TransformResponse(
        tokens=tokens,
        attention_weights=[aw.to_dict() for aw in attention_weights],
        layer_outputs=output.tolist(),
        latency_ms=latency_ms,
        metadata={
            "d_model": settings.d_model,
            "num_heads": settings.num_heads,
            "num_layers": settings.num_layers,
            "seq_length": len(tokens)
        }
    )

@app.websocket("/ws/transform")
async def websocket_transform(websocket: WebSocket):
    """
    WebSocket endpoint for real-time attention visualization.
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")
            
            if not text:
                continue
            
            # Tokenize
            token_ids = tokenizer.tokenize(text)
            tokens = [tokenizer.reverse_vocab.get(t, "<UNK>") for t in token_ids]
            
            # Create embeddings
            embeddings = tokenizer.create_embeddings(token_ids, settings.d_model)
            
            # Stream attention weights as they're computed
            await websocket.send_json({
                "type": "tokens",
                "data": {"tokens": tokens}
            })
            
            # Forward pass
            output, attention_weights = await transformer.forward(embeddings)
            
            # Send each layer's attention weights
            for aw in attention_weights:
                await websocket.send_json({
                    "type": "attention",
                    "data": aw.to_dict()
                })
                await asyncio.sleep(0.1)  # Simulate streaming
            
            # Send completion
            await websocket.send_json({
                "type": "complete",
                "data": {"latency_ms": 0}
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/config")
async def get_config():
    """Return current transformer configuration."""
    return {
        "d_model": settings.d_model,
        "num_heads": settings.num_heads,
        "num_layers": settings.num_layers,
        "dropout": settings.dropout,
        "max_seq_length": settings.max_seq_length
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
