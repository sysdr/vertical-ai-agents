#!/bin/bash

# L9: Implementing Agent Memory - Full Automated Setup
# Creates dual-memory system with short-term dict + long-term file storage

set -e

PROJECT_NAME="L9_Agent_Memory"
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"
DATA_DIR="data"

echo "========================================="
echo "L9: Agent Memory System Setup"
echo "Building on L8's decision_maker foundation"
echo "========================================="

# Create project structure
mkdir -p $PROJECT_NAME/{$BACKEND_DIR,$FRONTEND_DIR,$DATA_DIR/users,logs,tests}
cd $PROJECT_NAME

# ============================================
# BACKEND: Python FastAPI Memory Manager
# ============================================

cat > $BACKEND_DIR/memory_manager.py << 'EOF'
"""
Agent Memory Manager - Dual-Tier System
Short-term: In-memory dict for session context
Long-term: File-based JSON for persistent storage
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class MemoryManager:
    def __init__(self, data_dir: str = "data/users"):
        self.sessions: Dict[str, List[dict]] = {}  # Short-term memory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "short_term_hits": 0,
            "long_term_hits": 0,
            "stores": 0,
            "retrievals": 0
        }
    
    def store_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> dict:
        """Store message in short-term memory"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append(message)
        
        # Keep last 20 messages (working memory limit)
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
        
        self.metrics["stores"] += 1
        return message
    
    def recall_session(self, session_id: str, limit: int = 10) -> List[dict]:
        """Recall recent messages from short-term memory"""
        self.metrics["short_term_hits"] += 1
        self.metrics["retrievals"] += 1
        
        if session_id in self.sessions:
            return self.sessions[session_id][-limit:]
        return []
    
    def save_to_longterm(self, user_id: str, fact: dict) -> bool:
        """Persist fact to long-term storage"""
        try:
            filepath = self.data_dir / f"{user_id}.json"
            
            # Load existing data
            existing = []
            if filepath.exists():
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            
            # Append new fact
            entry = {
                "timestamp": datetime.now().isoformat(),
                "fact": fact
            }
            existing.append(entry)
            
            # Save with pretty printing
            with open(filepath, 'w') as f:
                json.dump(existing, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Long-term storage error: {e}")
            return False
    
    def search_longterm(self, user_id: str, keywords: List[str] = None, limit: int = 5) -> List[dict]:
        """Search long-term memory for relevant facts"""
        self.metrics["long_term_hits"] += 1
        self.metrics["retrievals"] += 1
        
        try:
            filepath = self.data_dir / f"{user_id}.json"
            
            if not filepath.exists():
                return []
            
            with open(filepath, 'r') as f:
                entries = json.load(f)
            
            # If no keywords, return recent entries
            if not keywords:
                return sorted(entries, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
            # Simple keyword matching
            matches = []
            for entry in entries:
                fact_text = json.dumps(entry['fact']).lower()
                if any(kw.lower() in fact_text for kw in keywords):
                    matches.append(entry)
            
            # Sort by recency
            matches = sorted(matches, key=lambda x: x['timestamp'], reverse=True)[:limit]
            return matches
            
        except Exception as e:
            print(f"Long-term search error: {e}")
            return []
    
    def get_metrics(self) -> dict:
        """Return memory system metrics"""
        return {
            **self.metrics,
            "active_sessions": len(self.sessions),
            "total_session_messages": sum(len(msgs) for msgs in self.sessions.values())
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear short-term memory for session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
EOF

cat > $BACKEND_DIR/api.py << 'EOF'
"""
FastAPI Memory API
Endpoints for storing and retrieving agent memory
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from memory_manager import MemoryManager

app = FastAPI(title="Agent Memory API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory manager
memory = MemoryManager()

# Request models
class MessageStore(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: Optional[dict] = None

class LongTermStore(BaseModel):
    user_id: str
    fact: dict

class SearchRequest(BaseModel):
    user_id: str
    keywords: Optional[List[str]] = None
    limit: int = 5

@app.post("/api/store")
async def store_message(request: MessageStore):
    """Store message in short-term memory"""
    message = memory.store_message(
        request.session_id,
        request.role,
        request.content,
        request.metadata
    )
    return {"success": True, "message": message}

@app.get("/api/recall/{session_id}")
async def recall_session(session_id: str, limit: int = 10):
    """Recall messages from short-term memory"""
    messages = memory.recall_session(session_id, limit)
    return {"success": True, "messages": messages, "count": len(messages)}

@app.post("/api/longterm/store")
async def store_longterm(request: LongTermStore):
    """Store fact in long-term memory"""
    success = memory.save_to_longterm(request.user_id, request.fact)
    if success:
        return {"success": True, "message": "Fact stored in long-term memory"}
    raise HTTPException(status_code=500, detail="Failed to store fact")

@app.post("/api/longterm/search")
async def search_longterm(request: SearchRequest):
    """Search long-term memory"""
    results = memory.search_longterm(
        request.user_id,
        request.keywords,
        request.limit
    )
    return {"success": True, "results": results, "count": len(results)}

@app.get("/api/metrics")
async def get_metrics():
    """Get memory system metrics"""
    metrics = memory.get_metrics()
    return {"success": True, "metrics": metrics}

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear short-term memory for session"""
    success = memory.clear_session(session_id)
    if success:
        return {"success": True, "message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Agent Memory API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

cat > $BACKEND_DIR/requirements.txt << 'EOF'
fastapi==0.115.5
uvicorn[standard]==0.32.1
pydantic==2.10.3
python-multipart==0.0.19
EOF

# ============================================
# FRONTEND: React Memory Dashboard
# ============================================

cat > $FRONTEND_DIR/package.json << 'EOF'
{
  "name": "agent-memory-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "axios": "^1.7.9",
    "core-js-pure": "^3.47.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "devDependencies": {
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
EOF

mkdir -p $FRONTEND_DIR/public $FRONTEND_DIR/src

cat > $FRONTEND_DIR/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Agent Memory Dashboard</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
EOF

cat > $FRONTEND_DIR/src/index.js << 'EOF'
import 'core-js-pure/features/global-this';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

cat > $FRONTEND_DIR/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [sessionId, setSessionId] = useState('session_001');
  const [userId, setUserId] = useState('user_001');
  const [message, setMessage] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [shortTermMemory, setShortTermMemory] = useState([]);
  const [longTermResults, setLongTermResults] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [factInput, setFactInput] = useState('');

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/metrics`);
      setMetrics(res.data.metrics);
    } catch (err) {
      console.error('Metrics error:', err);
    }
  };

  const storeMessage = async () => {
    if (!message.trim()) return;
    
    try {
      await axios.post(`${API_BASE}/api/store`, {
        session_id: sessionId,
        role: 'user',
        content: message,
        metadata: { source: 'dashboard' }
      });
      
      setMessage('');
      recallSession();
      loadMetrics();
    } catch (err) {
      alert('Error storing message: ' + err.message);
    }
  };

  const recallSession = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/recall/${sessionId}`);
      setShortTermMemory(res.data.messages);
    } catch (err) {
      console.error('Recall error:', err);
    }
  };

  const storeLongTermFact = async () => {
    if (!factInput.trim()) return;
    
    try {
      await axios.post(`${API_BASE}/api/longterm/store`, {
        user_id: userId,
        fact: { content: factInput, type: 'user_preference' }
      });
      
      setFactInput('');
      alert('Fact stored in long-term memory!');
      loadMetrics();
    } catch (err) {
      alert('Error storing fact: ' + err.message);
    }
  };

  const searchLongTerm = async () => {
    try {
      const keywords = searchQuery.split(' ').filter(k => k.length > 0);
      const res = await axios.post(`${API_BASE}/api/longterm/search`, {
        user_id: userId,
        keywords: keywords.length > 0 ? keywords : null,
        limit: 5
      });
      
      setLongTermResults(res.data.results);
      loadMetrics();
    } catch (err) {
      alert('Search error: ' + err.message);
    }
  };

  const clearSession = async () => {
    try {
      await axios.delete(`${API_BASE}/api/session/${sessionId}`);
      setShortTermMemory([]);
      alert('Session cleared!');
      loadMetrics();
    } catch (err) {
      alert('Clear error: ' + err.message);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🧠 Agent Memory System</h1>
        <p>L9: Dual-Tier Memory Architecture</p>
      </header>

      <div className="container">
        {/* Metrics Panel */}
        <div className="metrics-panel">
          <h2>📊 Memory Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Short-Term Hits</span>
              <span className="metric-value">{metrics.short_term_hits || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Long-Term Hits</span>
              <span className="metric-value">{metrics.long_term_hits || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Active Sessions</span>
              <span className="metric-value">{metrics.active_sessions || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Total Stores</span>
              <span className="metric-value">{metrics.stores || 0}</span>
            </div>
          </div>
        </div>

        <div className="two-column">
          {/* Short-Term Memory */}
          <div className="panel">
            <h2>💬 Short-Term Memory</h2>
            <div className="config">
              <label>Session ID:</label>
              <input 
                type="text" 
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="session_001"
              />
            </div>
            
            <div className="message-input">
              <input 
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && storeMessage()}
                placeholder="Enter a message..."
              />
              <button onClick={storeMessage}>Store</button>
              <button onClick={recallSession} className="secondary">Recall</button>
              <button onClick={clearSession} className="danger">Clear</button>
            </div>

            <div className="memory-list">
              {shortTermMemory.length === 0 ? (
                <p className="empty">No messages in session</p>
              ) : (
                shortTermMemory.map((msg, idx) => (
                  <div key={idx} className="memory-item">
                    <span className="role">{msg.role}</span>
                    <span className="content">{msg.content}</span>
                    <span className="timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Long-Term Memory */}
          <div className="panel">
            <h2>💾 Long-Term Memory</h2>
            <div className="config">
              <label>User ID:</label>
              <input 
                type="text" 
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="user_001"
              />
            </div>

            <div className="fact-input">
              <input 
                type="text"
                value={factInput}
                onChange={(e) => setFactInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && storeLongTermFact()}
                placeholder="Store a fact (e.g., 'Likes pizza')"
              />
              <button onClick={storeLongTermFact}>Store Fact</button>
            </div>

            <div className="search-input">
              <input 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchLongTerm()}
                placeholder="Search keywords..."
              />
              <button onClick={searchLongTerm}>Search</button>
            </div>

            <div className="memory-list">
              {longTermResults.length === 0 ? (
                <p className="empty">No results found</p>
              ) : (
                longTermResults.map((entry, idx) => (
                  <div key={idx} className="memory-item">
                    <span className="content">{JSON.stringify(entry.fact)}</span>
                    <span className="timestamp">{new Date(entry.timestamp).toLocaleDateString()}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
EOF

cat > $FRONTEND_DIR/src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  padding: 20px;
}

.header {
  text-align: center;
  color: white;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
}

.header p {
  font-size: 1.1rem;
  opacity: 0.9;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
}

.metrics-panel {
  background: white;
  border-radius: 15px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.metrics-panel h2 {
  margin-bottom: 20px;
  color: #333;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.metric-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  border-radius: 10px;
  color: white;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.metric-label {
  font-size: 0.9rem;
  opacity: 0.9;
}

.metric-value {
  font-size: 2rem;
  font-weight: bold;
}

.two-column {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.panel {
  background: white;
  border-radius: 15px;
  padding: 25px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.panel h2 {
  margin-bottom: 20px;
  color: #333;
}

.config {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  align-items: center;
}

.config label {
  font-weight: 600;
  color: #555;
}

.config input {
  flex: 1;
  padding: 10px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 0.95rem;
}

.message-input, .fact-input, .search-input {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.message-input input, .fact-input input, .search-input input {
  flex: 1;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
}

button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
}

button:hover {
  transform: translateY(-2px);
}

button.secondary {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

button.danger {
  background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
}

.memory-list {
  max-height: 400px;
  overflow-y: auto;
  border: 2px solid #f0f0f0;
  border-radius: 10px;
  padding: 15px;
}

.empty {
  text-align: center;
  color: #999;
  padding: 20px;
}

.memory-item {
  background: #f8f9fa;
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 10px;
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.role {
  font-weight: 600;
  color: #667eea;
  font-size: 0.85rem;
  text-transform: uppercase;
}

.content {
  color: #333;
  font-size: 0.95rem;
}

.timestamp {
  color: #999;
  font-size: 0.8rem;
}

@media (max-width: 968px) {
  .two-column {
    grid-template-columns: 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr 1fr;
  }
}
EOF

# ============================================
# DOCKER CONFIGURATION
# ============================================

cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create data directory
RUN mkdir -p data/users

EXPOSE 8000

CMD ["python", "backend/api.py"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./backend:/app/backend
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  frontend:
    image: node:20-alpine
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "3000:3000"
    command: sh -c "npm install && npm start"
    depends_on:
      - backend
    restart: unless-stopped
EOF

# ============================================
# HELPER SCRIPTS
# ============================================

cat > build.sh << 'EOF'
#!/bin/bash
echo "Building L9 Agent Memory System..."

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
pip install -r requirements.txt
cd ..

# Frontend setup
cd frontend
npm install
cd ..

echo "✅ Build complete!"
EOF

cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting L9 Agent Memory System..."

# Start backend
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
python api.py &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "✅ System running!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Dashboard: http://localhost:3000"
echo "API: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

wait
EOF

cat > stop.sh << 'EOF'
#!/bin/bash
echo "Stopping L9 Agent Memory System..."

pkill -f "python api.py"
pkill -f "react-scripts start"

echo "✅ Stopped!"
EOF

cat > test.sh << 'EOF'
#!/bin/bash
echo "Testing L9 Memory System..."

# Check backend health
echo "1. Checking backend health..."
curl -s http://localhost:8000/health | grep -q "healthy"
if [ $? -eq 0 ]; then
    echo "   ✅ Backend healthy"
else
    echo "   ❌ Backend not responding"
    exit 1
fi

# Test short-term storage
echo "2. Testing short-term memory storage..."
curl -s -X POST http://localhost:8000/api/store \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test_session","role":"user","content":"I like pizza"}' | grep -q "success"
if [ $? -eq 0 ]; then
    echo "   ✅ Short-term storage works"
else
    echo "   ❌ Short-term storage failed"
    exit 1
fi

# Test short-term recall
echo "3. Testing short-term memory recall..."
curl -s http://localhost:8000/api/recall/test_session | grep -q "pizza"
if [ $? -eq 0 ]; then
    echo "   ✅ Short-term recall works"
else
    echo "   ❌ Short-term recall failed"
    exit 1
fi

# Test long-term storage
echo "4. Testing long-term memory storage..."
curl -s -X POST http://localhost:8000/api/longterm/store \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","fact":{"preference":"pizza","type":"food"}}' | grep -q "success"
if [ $? -eq 0 ]; then
    echo "   ✅ Long-term storage works"
else
    echo "   ❌ Long-term storage failed"
    exit 1
fi

# Test long-term search
echo "5. Testing long-term memory search..."
curl -s -X POST http://localhost:8000/api/longterm/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","keywords":["pizza"],"limit":5}' | grep -q "preference"
if [ $? -eq 0 ]; then
    echo "   ✅ Long-term search works"
else
    echo "   ❌ Long-term search failed"
    exit 1
fi

# Test metrics
echo "6. Testing metrics endpoint..."
curl -s http://localhost:8000/api/metrics | grep -q "short_term_hits"
if [ $? -eq 0 ]; then
    echo "   ✅ Metrics endpoint works"
else
    echo "   ❌ Metrics endpoint failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ All tests passed!"
echo "========================================="
echo "Open http://localhost:3000 to see dashboard"
EOF

chmod +x build.sh start.sh stop.sh test.sh

# ============================================
# README
# ============================================

cat > README.md << 'EOF'
# L9: Agent Memory Implementation

## Overview
Dual-tier memory system for AI agents:
- **Short-term**: In-memory dict for session context (fast)
- **Long-term**: File-based JSON for persistent storage

## Quick Start

### Docker (Recommended)
```bash
docker-compose up --build
```

### Local Development
```bash
./build.sh    # Install dependencies
./start.sh    # Start backend + frontend
./test.sh     # Run verification tests
```

## Access Points
- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Architecture

### Short-Term Memory
- Session-scoped message history
- Last 20 messages retained
- In-memory dict (< 10ms retrieval)

### Long-Term Memory
- User-scoped persistent facts
- JSON file storage
- Keyword search capability

## API Endpoints

### Store Message
```bash
POST /api/store
{
  "session_id": "session_001",
  "role": "user",
  "content": "I like pizza"
}
```

### Recall Session
```bash
GET /api/recall/{session_id}?limit=10
```

### Store Long-Term Fact
```bash
POST /api/longterm/store
{
  "user_id": "user_001",
  "fact": {"preference": "pizza"}
}
```

### Search Long-Term
```bash
POST /api/longterm/search
{
  "user_id": "user_001",
  "keywords": ["pizza"],
  "limit": 5
}
```

## Production Notes
- Replace file storage with Redis (short-term) + PostgreSQL (long-term)
- Add vector embeddings for semantic search
- Implement async writes for high throughput
- Add memory eviction policies
- Monitor memory usage per session

## Next Steps
L10 will combine this memory system with L8's decision_maker to build SimpleAgent class.
EOF

echo ""
echo "========================================="
echo "✅ L9 Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. ./build.sh"
echo "  3. ./start.sh"
echo "  4. Open http://localhost:3000"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
echo ""