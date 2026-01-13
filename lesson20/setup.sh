#!/bin/bash
set -e

PROJECT_NAME="l20-vector-databases"
# GEMINI_API_KEY should be set via environment variable or .env file

echo "🚀 Setting up L20: Vector Databases & Indexing"

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,data/chromadb,tests}
cd $PROJECT_NAME

# Backend: requirements.txt
cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
chromadb==0.4.22
google-generativeai==0.4.0
pydantic==2.5.3
python-multipart==0.0.6
aiofiles==23.2.1
websockets==12.0
pytest==7.4.4
httpx==0.26.0
numpy<2.0.0
EOF

# Backend: vector_store.py
cat > backend/vector_store.py << 'EOF'
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
EOF

# Backend: embedding_service.py
cat > backend/embedding_service.py << 'EOF'
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
        """Embed texts with batching for rate limit efficiency"""
        embeddings = []
        batch_size = 100  # Gemini's max batch size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                result = await asyncio.to_thread(
                    genai.embed_content,
                    model=self.model,
                    content=batch,
                    task_type=task_type
                )
                # Handle batch response: result['embeddings'] is a list of dicts with 'values' key
                if isinstance(result, dict):
                    if 'embeddings' in result:
                        # Batch response: list of embedding dicts
                        batch_embeddings = [emb['values'] for emb in result['embeddings']]
                        embeddings.extend(batch_embeddings)
                    elif 'embedding' in result:
                        # Single response (shouldn't happen with batch, but handle it)
                        if 'values' in result['embedding']:
                            embeddings.append(result['embedding']['values'])
                        else:
                            embeddings.append(result['embedding'])
                    else:
                        # Fallback: try direct access
                        logger.warning(f"Unexpected response format: {result.keys()}")
                        if 'values' in result:
                            embeddings.append(result['values'])
                
                logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch)} texts)")
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                # Return zero vectors on failure (768 is text-embedding-004 dimension)
                embeddings.extend([[0.0] * 768 for _ in batch])
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
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
        return result.get('values', [])
EOF

# Backend: indexing_pipeline.py
cat > backend/indexing_pipeline.py << 'EOF'
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
EOF

# Backend: search_engine.py
cat > backend/search_engine.py << 'EOF'
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def search(
        self,
        query: str,
        collection_name: str = "default",
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search with optional metadata filtering"""
        # Embed query
        query_embedding = await self.embedding_service.embed_query(query)
        
        # Search vector store
        raw_results = self.vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=n_results,
            where=filters
        )
        
        # Format results
        results = []
        for i in range(len(raw_results['ids'][0])):
            results.append({
                "id": raw_results['ids'][0][i],
                "document": raw_results['documents'][0][i],
                "metadata": raw_results['metadatas'][0][i],
                "distance": raw_results['distances'][0][i],
                "similarity": 1 - raw_results['distances'][0][i]  # Cosine similarity
            })
        
        logger.info(f"Found {len(results)} results for query: {query[:50]}")
        return results
EOF

# Backend: main.py
cat > backend/main.py << 'EOF'
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")
    
    # Send initial stats
    try:
        stats = vector_store.get_stats("default")
        await websocket.send_json(stats)
    except Exception as e:
        logger.error(f"Error sending initial stats: {e}")
    
    try:
        while True:
            # Keep connection alive by receiving messages (ping/pong)
            try:
                data = await websocket.receive_text()
                # Echo back or handle client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
            except Exception as e:
                logger.info(f"WebSocket receive error: {e}")
                break
    except Exception as e:
        logger.info(f"WebSocket connection error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_connections)}")

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
EOF

# Frontend: package.json
cat > frontend/package.json << 'EOF'
{
  "name": "l20-vector-db-dashboard",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "recharts": "^2.10.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.0"
  }
}
EOF

# Frontend: vite.config.js
cat > frontend/vite.config.js << 'EOF'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 3000 }
})
EOF

# Frontend: index.html
cat > frontend/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>L20: Vector Database Dashboard</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
EOF

# Frontend: main.jsx
cat > frontend/src/main.jsx << 'EOF'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
EOF

# Frontend: App.jsx
cat > frontend/src/App.jsx << 'EOF'
import React, { useState, useEffect } from 'react';
import IndexPanel from './components/IndexPanel';
import SearchPanel from './components/SearchPanel';
import StatsPanel from './components/StatsPanel';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState({ total_documents: 0 });
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError] = useState(null);
  const [indexError, setIndexError] = useState(null);
  const [indexSuccess, setIndexSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // WebSocket for real-time stats
    let ws;
    let reconnectTimeout;
    let pingInterval;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = () => {
          console.log('WebSocket connected');
          // Clear any reconnect timeout
          if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
          }
          // Set up ping to keep connection alive
          pingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, 30000); // Ping every 30 seconds
        };
        
        ws.onmessage = (event) => {
          try {
            // Ignore pong messages
            if (event.data === 'pong') {
              return;
            }
            const data = JSON.parse(event.data);
            setStats(data);
          } catch (e) {
            console.error('WebSocket message parse error:', e);
          }
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
        
        ws.onclose = (event) => {
          console.log('WebSocket closed', event.code, event.reason);
          // Clear ping interval
          if (pingInterval) {
            clearInterval(pingInterval);
            pingInterval = null;
          }
          // Attempt to reconnect after 3 seconds
          if (!reconnectTimeout) {
            reconnectTimeout = setTimeout(() => {
              console.log('Attempting to reconnect WebSocket...');
              connectWebSocket();
            }, 3000);
          }
        };
      } catch (error) {
        console.error('WebSocket connection error:', error);
        // Retry connection after error
        reconnectTimeout = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      }
    };

    // Initial connection
    connectWebSocket();

    // Initial stats fetch (fallback if WebSocket fails)
    fetchStats();

    return () => {
      if (pingInterval) {
        clearInterval(pingInterval);
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats/default`);
      if (!response.ok) {
        throw new Error('Failed to fetch stats');
      }
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Stats fetch error:', error);
    }
  };

  const handleIndex = async (text, source) => {
    setIndexError(null);
    setIndexSuccess(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source, collection: 'default' })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Indexing failed');
      }
      const result = await response.json();
      setIndexSuccess(`Successfully indexed ${result.chunks} chunks from ${result.source}`);
      await fetchStats();
      // Clear success message after 3 seconds
      setTimeout(() => setIndexSuccess(null), 3000);
    } catch (error) {
      console.error('Indexing error:', error);
      setIndexError(error.message || 'Failed to index document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    setSearchError(null);
    setSearchResults([]);
    if (!query.trim()) {
      setSearchError('Please enter a search query');
      return;
    }
    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, collection: 'default', n_results: 5 })
      });
      if (!response.ok) {
        const error = await response.json();
        console.error('Search error:', error);
        const errorMessage = error.detail || 'Search failed. Please try again.';
        setSearchError(errorMessage);
        setSearchResults([]);
        return;
      }
      const data = await response.json();
      setSearchResults(data.results || []);
      if (data.results && data.results.length === 0) {
        setSearchError('No results found. Try a different query or index some documents first.');
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchError('Network error. Please check your connection and try again.');
      setSearchResults([]);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🗄️ L20: Vector Database Dashboard</h1>
        <p>ChromaDB + Gemini Embeddings - Production Vector Storage</p>
      </header>

      <div className="container">
        <StatsPanel stats={stats} />
        <IndexPanel onIndex={handleIndex} error={indexError} success={indexSuccess} loading={loading} />
        <SearchPanel onSearch={handleSearch} results={searchResults} error={searchError} />
      </div>
    </div>
  );
}

export default App;
EOF

# Frontend: IndexPanel.jsx
cat > frontend/src/components/IndexPanel.jsx << 'EOF'
import React, { useState } from 'react';

function IndexPanel({ onIndex }) {
  const [text, setText] = useState('');
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text || !source) return;

    setLoading(true);
    await onIndex(text, source);
    setText('');
    setSource('');
    setLoading(false);
  };

  return (
    <div className="panel">
      <h2>📄 Index Documents</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Source (e.g., doc1.pdf)"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        />
        <textarea
          placeholder="Document text to index..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Indexing...' : 'Index Document'}
        </button>
      </form>
    </div>
  );
}

export default IndexPanel;
EOF

# Frontend: SearchPanel.jsx
cat > frontend/src/components/SearchPanel.jsx << 'EOF'
import React, { useState } from 'react';

function SearchPanel({ onSearch, results, error }) {
  const [query, setQuery] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (query) onSearch(query);
  };

  return (
    <div className="panel">
      <h2>🔍 Semantic Search</h2>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Enter search query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>

      <div className="results">
        {results && results.length > 0 ? (
          results.map((result, idx) => (
            <div key={idx} className="result-card">
              <div className="result-header">
                <span className="similarity">
                  {(result.similarity * 100).toFixed(1)}% match
                </span>
                <span className="source">{result.metadata?.source || 'Unknown'}</span>
              </div>
              <p className="result-text">{result.document}</p>
              <div className="result-meta">
                Chunk {result.metadata?.chunk_index + 1}/{result.metadata?.chunk_total}
              </div>
            </div>
          ))
        ) : (
          <p style={{ padding: '1rem', color: '#666', textAlign: 'center' }}>
            No results found. Try searching with a different query.
          </p>
        )}
      </div>
    </div>
  );
}

export default SearchPanel;
EOF

# Frontend: StatsPanel.jsx
cat > frontend/src/components/StatsPanel.jsx << 'EOF'
import React from 'react';

function StatsPanel({ stats }) {
  return (
    <div className="panel stats-panel">
      <h2>📊 Collection Stats</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats.total_documents || 0}</h3>
          <p>Indexed Documents</p>
        </div>
        <div className="stat-card">
          <h3>{stats.index_type || 'HNSW'}</h3>
          <p>Index Algorithm</p>
        </div>
        <div className="stat-card">
          <h3>{stats.distance_metric || 'Cosine'}</h3>
          <p>Distance Metric</p>
        </div>
      </div>
    </div>
  );
}

export default StatsPanel;
EOF

# Frontend: App.css
cat > frontend/src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.app {
  padding: 2rem;
}

header {
  text-align: center;
  color: white;
  margin-bottom: 2rem;
}

header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
}

.panel {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.panel h2 {
  margin-bottom: 1rem;
  color: #333;
}

form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

input, textarea {
  padding: 0.75rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
}

button {
  padding: 0.75rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: transform 0.2s;
}

button:hover {
  transform: translateY(-2px);
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.stat-card {
  text-align: center;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-card h3 {
  font-size: 2rem;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.results {
  margin-top: 1rem;
  max-height: 500px;
  overflow-y: auto;
}

.result-card {
  padding: 1rem;
  margin-bottom: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #667eea;
}

.result-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.similarity {
  color: #667eea;
  font-weight: bold;
}

.source {
  color: #666;
}

.result-text {
  margin: 0.5rem 0;
  line-height: 1.5;
}

.result-meta {
  font-size: 0.85rem;
  color: #999;
}
EOF

# Docker files
cat > Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
RUN mkdir -p /app/data/chromadb

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > Dockerfile.frontend << 'EOF'
FROM node:20-alpine

WORKDIR /app
COPY frontend/package.json .
RUN npm install

COPY frontend/ .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host"]
EOF

cat > docker-compose.yml << 'EOF'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./data/chromadb:/app/data/chromadb
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
EOF

# Helper scripts
cat > build.sh << 'EOF'
#!/bin/bash
if command -v docker &> /dev/null; then
    echo "🐳 Building Docker containers..."
    docker-compose build
else
    echo "📦 Installing dependencies..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r backend/requirements.txt
    cd frontend && npm install && cd ..
fi
echo "✅ Build complete"
EOF

cat > start.sh << 'EOF'
#!/bin/bash
if command -v docker &> /dev/null; then
    echo "🐳 Starting Docker containers..."
    docker-compose up -d
    echo "✅ Services running:"
    echo "   Backend: http://localhost:8000"
    echo "   Frontend: http://localhost:3000"
else
    echo "🚀 Starting services..."
    source venv/bin/activate
    cd backend && uvicorn main:app --reload --port 8000 &
    cd frontend && npm run dev &
    echo "✅ Services started"
fi
EOF

cat > stop.sh << 'EOF'
#!/bin/bash
if command -v docker &> /dev/null; then
    echo "🛑 Stopping Docker containers..."
    docker-compose down
else
    echo "🛑 Stopping services..."
    pkill -f uvicorn
    pkill -f vite
fi
echo "✅ Services stopped"
EOF

cat > test.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing L20 Vector Database..."

# Test indexing
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text":"Machine learning is a subset of artificial intelligence focused on algorithms that learn from data.","source":"ml-basics.txt","collection":"default"}'

# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","collection":"default","n_results":3}'

# Test stats
curl http://localhost:8000/stats/default

echo -e "\n✅ Tests complete - Check http://localhost:3000 for dashboard"
EOF

chmod +x build.sh start.sh stop.sh test.sh

# README
cat > README.md << 'EOF'
# L20: Vector Databases & Indexing

## Quick Start
```bash
./build.sh   # Install dependencies
./start.sh   # Start services
./test.sh    # Run tests
```

Dashboard: http://localhost:3000
API: http://localhost:8000/docs

## Architecture

- **Backend**: FastAPI + ChromaDB + Gemini Embeddings
- **Frontend**: React + Real-time WebSocket updates
- **Storage**: Persistent ChromaDB with HNSW indexing

## Features

- Character-based document chunking
- Batch embedding with Gemini API
- Semantic search with metadata filtering
- Real-time collection statistics
- Production-ready error handling
EOF

# Validate all files were generated
echo ""
echo "🔍 Validating generated files..."

declare -a REQUIRED_FILES=(
    "backend/requirements.txt"
    "backend/vector_store.py"
    "backend/embedding_service.py"
    "backend/indexing_pipeline.py"
    "backend/search_engine.py"
    "backend/main.py"
    "frontend/package.json"
    "frontend/vite.config.js"
    "frontend/index.html"
    "frontend/src/main.jsx"
    "frontend/src/App.jsx"
    "frontend/src/components/IndexPanel.jsx"
    "frontend/src/components/SearchPanel.jsx"
    "frontend/src/components/StatsPanel.jsx"
    "frontend/src/App.css"
    "Dockerfile.backend"
    "Dockerfile.frontend"
    "docker-compose.yml"
    "build.sh"
    "start.sh"
    "stop.sh"
    "test.sh"
    "README.md"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Missing files detected:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    cd ..
    exit 1
fi

cd ..
echo "✅ All files generated successfully!"
echo ""
echo "✅ L20 setup complete!"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  ./build.sh"
echo "  ./start.sh"
echo "  Open http://localhost:3000"
echo ""