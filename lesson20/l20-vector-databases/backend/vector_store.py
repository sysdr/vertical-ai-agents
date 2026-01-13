import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chromadb"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collections: Dict[str, chromadb.Collection] = {}
        logger.info(f"ChromaDB initialized at {persist_directory}")
    
    def get_or_create_collection(self, name: str = "default") -> chromadb.Collection:
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{name}' ready")
        return self.collections[name]
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_stats(self, collection_name: str = "default") -> Dict[str, Any]:
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        return {
            "collection": collection_name,
            "total_documents": count,
            "index_type": "HNSW",
            "distance_metric": "cosine"
        }
    
    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]
    
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
        if name in self.collections:
            del self.collections[name]
        logger.info(f"Deleted collection '{name}'")
