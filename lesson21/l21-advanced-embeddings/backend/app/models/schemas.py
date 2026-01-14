from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChunkRequest(BaseModel):
    text: str
    strategy: str = "semantic"
    params: Dict[str, Any] = Field(default_factory=dict)

class ChunkInfo(BaseModel):
    text: str
    index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkResponse(BaseModel):
    strategy: str
    chunks: List[Dict[str, Any]]
    total_chunks: int
    metadata: Dict[str, Any]

class EmbedRequest(BaseModel):
    texts: List[str]
    model: str = "sentence-transformer"

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    dimension: int
    metadata: Dict[str, Any]

class CompareRequest(BaseModel):
    text: str
    strategies: List[str]
    models: List[str]
    params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class CompareResponse(BaseModel):
    results: Dict[str, Dict[str, Any]]
    total_comparisons: int

class DocumentUpload(BaseModel):
    filename: str
    content: str
    strategy: str = "semantic"
