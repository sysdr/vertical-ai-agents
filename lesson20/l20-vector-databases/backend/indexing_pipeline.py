from typing import List, Dict, Any
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class IndexingPipeline:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Character-based chunking (reused from L19)"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks
    
    async def index_document(
        self,
        document_text: str,
        source: str,
        collection_name: str = "default",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Full indexing pipeline: chunk → embed → store"""
        # Step 1: Chunk
        chunks = self.chunk_document(document_text)
        logger.info(f"Chunked document into {len(chunks)} pieces")
        
        # Step 2: Embed
        embeddings = await self.embedding_service.embed_texts(chunks)
        
        # Step 3: Prepare metadata
        base_metadata = metadata or {}
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}_{i}_{chunk[:50]}".encode()).hexdigest()
            ids.append(chunk_id)
            
            chunk_metadata = {
                **base_metadata,
                "source": source,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "indexed_at": datetime.utcnow().isoformat(),
                "chunking_strategy": "character",
                "chunk_size": len(chunk)
            }
            metadatas.append(chunk_metadata)
        
        # Step 4: Store
        self.vector_store.add_documents(
            collection_name=collection_name,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "status": "indexed",
            "chunks": len(chunks),
            "collection": collection_name,
            "source": source
        }
