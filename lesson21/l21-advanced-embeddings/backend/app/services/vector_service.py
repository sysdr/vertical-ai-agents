try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None

from typing import List, Dict
import uuid

class VectorService:
    def __init__(self):
        self.client = None
        self.collections = {}
    
    async def initialize(self):
        """Initialize ChromaDB client (builds on L20)"""
        if not CHROMADB_AVAILABLE:
            print("Warning: chromadb is not installed. Vector storage will be disabled.")
            return
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./data/chromadb"
            ))
            
            # Create collections for different strategies
            strategies = ["fixed", "semantic", "recursive", "sliding_window", "sentence"]
            for strategy in strategies:
                collection_name = f"l21_{strategy}"
                self.collections[strategy] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"lesson": "L21", "strategy": strategy}
                )
        except Exception as e:
            print(f"Warning: Could not initialize ChromaDB: {e}")
    
    async def store_document(self, document_id: str, chunks: List[Dict], strategy: str):
        """Store chunked document in ChromaDB"""
        if not CHROMADB_AVAILABLE or self.client is None:
            raise ValueError("ChromaDB is not available. Please install chromadb.")
        collection = self.collections.get(strategy)
        if not collection:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Prepare data for ChromaDB
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk.get('metadata', {}) for chunk in chunks]
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return document_id
    
    async def list_collections(self) -> List[str]:
        """List available collections"""
        if not CHROMADB_AVAILABLE or self.client is None:
            return []
        return list(self.collections.keys())
