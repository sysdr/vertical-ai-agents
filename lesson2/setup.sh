#!/bin/bash
set -e

# L2: Advanced Python & Libraries - Full Implementation
# Builds on L1's venv, adds async patterns, decorators, Pydantic validation
# Prepares foundation for L3's transformer architecture

echo "=== VAIA L2: Advanced Python & Libraries Setup ==="

# ============================================================================
# Project Structure
# ============================================================================
mkdir -p vaia-l2-advanced-python
cd vaia-l2-advanced-python

mkdir -p backend/app
mkdir -p frontend/src/components
mkdir -p frontend/public
mkdir -p tests
mkdir -p scripts

# ============================================================================
# Backend Implementation
# ============================================================================

# Main FastAPI Application
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
import time
from functools import wraps
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VAIA L2: Advanced Python Patterns")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models - Type-Safe Data Contracts
# ============================================================================

class AgentRequest(BaseModel):
    """Type-safe request model with validation"""
    agent_id: str = Field(..., regex=r'^agent-[a-z0-9]{8}$')
    prompts: List[str] = Field(..., min_items=1, max_items=100)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if any(len(p) > 5000 for p in v):
            raise ValueError("Individual prompts must be under 5000 characters")
        return v

class AgentResponse(BaseModel):
    """Type-safe response model"""
    agent_id: str
    results: List[str]
    latency_ms: float
    timestamp: str
    cached: bool = False

class SystemMetrics(BaseModel):
    """System performance metrics"""
    total_requests: int
    active_tasks: int
    avg_latency_ms: float
    success_rate: float
    cache_hit_rate: float
    timestamp: str

# ============================================================================
# Production Decorators
# ============================================================================

# Global metrics
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "cache_hits": 0,
    "cache_misses": 0
}

# Simple in-memory cache
response_cache: Dict[str, Any] = {}

def with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator: Automatic retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Max retries exceeded: {str(e)}")
                        raise
                    wait_time = backoff ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator

def with_cache(ttl_seconds: int = 60):
    """Decorator: Simple caching layer"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function args
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            if cache_key in response_cache:
                cached_data, cached_time = response_cache[cache_key]
                if time.time() - cached_time < ttl_seconds:
                    metrics["cache_hits"] += 1
                    logger.info(f"Cache hit: {cache_key[:50]}")
                    return cached_data, True  # Return data and cache status
            
            # Cache miss - execute function
            metrics["cache_misses"] += 1
            result = await func(*args, **kwargs)
            response_cache[cache_key] = (result, time.time())
            return result, False  # Return data and cache status
        return wrapper
    return decorator

def with_performance_tracking(func):
    """Decorator: Track execution time and success rate"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        metrics["total_requests"] += 1
        
        try:
            result = await func(*args, **kwargs)
            metrics["successful_requests"] += 1
            return result
        except Exception as e:
            metrics["failed_requests"] += 1
            raise
        finally:
            latency = (time.time() - start_time) * 1000
            metrics["total_latency"] += latency
            logger.info(f"Request completed in {latency:.2f}ms")
    
    return wrapper

# ============================================================================
# Async Business Logic
# ============================================================================

GEMINI_API_KEY = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

@with_retry(max_attempts=3, backoff=2.0)
async def call_gemini_api(session: aiohttp.ClientSession, prompt: str, temperature: float, max_tokens: int) -> str:
    """Async Gemini API call with retry logic"""
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        async with session.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=response.status, detail=f"Gemini API error: {error_text}")
            
            data = await response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Gemini API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {str(e)}")

@with_cache(ttl_seconds=300)
@with_performance_tracking
async def process_prompts_async(prompts: List[str], temperature: float, max_tokens: int) -> List[str]:
    """Process multiple prompts concurrently with caching"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            call_gemini_api(session, prompt, temperature, max_tokens)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Prompt {i} failed: {str(result)}")
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)
        
        return processed_results

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/process", response_model=AgentResponse)
async def process_agent_request(request: AgentRequest):
    """
    Process AI agent requests asynchronously with validation
    Demonstrates: async/await, Pydantic validation, decorators
    """
    start_time = time.time()
    
    try:
        # Process prompts with caching and retry logic
        results, cached = await process_prompts_async(
            request.prompts,
            request.temperature,
            request.max_tokens
        )
        
        latency = (time.time() - start_time) * 1000
        
        return AgentResponse(
            agent_id=request.agent_id,
            results=results,
            latency_ms=round(latency, 2),
            timestamp=datetime.now().isoformat(),
            cached=cached
        )
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get real-time system performance metrics"""
    total_reqs = metrics["total_requests"]
    successful = metrics["successful_requests"]
    
    return SystemMetrics(
        total_requests=total_reqs,
        active_tasks=len(asyncio.all_tasks()),
        avg_latency_ms=round(metrics["total_latency"] / max(total_reqs, 1), 2),
        success_rate=round(successful / max(total_reqs, 1), 3),
        cache_hit_rate=round(
            metrics["cache_hits"] / max(metrics["cache_hits"] + metrics["cache_misses"], 1),
            3
        ),
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "async_enabled": True,
        "decorators_active": True,
        "validation_enabled": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Requirements
cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
aiohttp==3.10.0
python-multipart==0.0.12
EOF

# ============================================================================
# Frontend Implementation
# ============================================================================

cat > frontend/package.json << 'EOF'
{
  "name": "vaia-l2-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "devDependencies": {
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [">0.2%", "not dead"],
    "development": ["last 1 chrome version"]
  }
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VAIA L2: Advanced Python Patterns</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;
EOF

cat > frontend/src/App.css << 'EOF'
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
EOF

cat > frontend/src/components/Dashboard.js << 'EOF'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE = 'http://localhost:8000';

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [agentId, setAgentId] = useState('agent-12345678');
  const [prompts, setPrompts] = useState('Explain async/await in Python\nWhat are decorators?\nHow does Pydantic work?');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await axios.get(`${API_BASE}/metrics`);
        setMetrics(res.data);
      } catch (err) {
        console.error('Metrics fetch error:', err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const promptList = prompts.split('\n').filter(p => p.trim());
      const res = await axios.post(`${API_BASE}/process`, {
        agent_id: agentId,
        prompts: promptList,
        temperature: parseFloat(temperature),
        max_tokens: parseInt(maxTokens)
      });
      setResponse(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🚀 VAIA L2: Advanced Python Patterns</h1>
        <p>Async Processing • Decorators • Pydantic Validation</p>
      </header>

      {metrics && (
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{metrics.total_requests}</div>
            <div className="metric-label">Total Requests</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.active_tasks}</div>
            <div className="metric-label">Active Tasks</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.avg_latency_ms}ms</div>
            <div className="metric-label">Avg Latency</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(metrics.success_rate * 100).toFixed(1)}%</div>
            <div className="metric-label">Success Rate</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(metrics.cache_hit_rate * 100).toFixed(1)}%</div>
            <div className="metric-label">Cache Hit Rate</div>
          </div>
        </div>
      )}

      <div className="content-grid">
        <div className="input-section">
          <h2>Agent Request</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Agent ID</label>
              <input
                type="text"
                value={agentId}
                onChange={(e) => setAgentId(e.target.value)}
                pattern="^agent-[a-z0-9]{8}$"
                placeholder="agent-12345678"
              />
            </div>

            <div className="form-group">
              <label>Prompts (one per line)</label>
              <textarea
                value={prompts}
                onChange={(e) => setPrompts(e.target.value)}
                rows="6"
                placeholder="Enter prompts..."
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                  min="1"
                  max="8192"
                />
              </div>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? '⏳ Processing...' : '▶️ Process Async'}
            </button>
          </form>
        </div>

        <div className="output-section">
          <h2>Response</h2>
          
          {error && (
            <div className="error-box">
              <strong>❌ Error:</strong> {error}
            </div>
          )}

          {response && (
            <div className="response-box">
              <div className="response-header">
                <span className={response.cached ? 'badge cached' : 'badge fresh'}>
                  {response.cached ? '💾 Cached' : '🔥 Fresh'}
                </span>
                <span className="latency">{response.latency_ms}ms</span>
              </div>
              
              <div className="results">
                {response.results.map((result, idx) => (
                  <div key={idx} className="result-item">
                    <div className="result-header">Result {idx + 1}</div>
                    <div className="result-text">{result}</div>
                  </div>
                ))}
              </div>
              
              <div className="response-footer">
                <small>Agent: {response.agent_id} • {response.timestamp}</small>
              </div>
            </div>
          )}

          {!response && !error && !loading && (
            <div className="placeholder">
              <p>✨ Submit a request to see async processing in action</p>
              <ul>
                <li>Multiple prompts processed concurrently</li>
                <li>Automatic retries with exponential backoff</li>
                <li>Response caching for efficiency</li>
                <li>Type-safe validation with Pydantic</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      <footer className="footer">
        <p>Building on L1's foundation → Preparing for L3's Transformer architecture</p>
      </footer>
    </div>
  );
}

export default Dashboard;
EOF

cat > frontend/src/components/Dashboard.css << 'EOF'
.dashboard {
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  background: white;
  padding: 30px;
  border-radius: 12px;
  margin-bottom: 30px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  text-align: center;
}

.header h1 {
  color: #2d3748;
  margin-bottom: 10px;
  font-size: 2.5em;
}

.header p {
  color: #718096;
  font-size: 1.1em;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.metric-card {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  text-align: center;
  transition: transform 0.2s;
}

.metric-card:hover {
  transform: translateY(-5px);
}

.metric-value {
  font-size: 2.5em;
  font-weight: bold;
  color: #667eea;
  margin-bottom: 10px;
}

.metric-label {
  color: #718096;
  font-size: 0.9em;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  margin-bottom: 30px;
}

@media (max-width: 968px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}

.input-section, .output-section {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.input-section h2, .output-section h2 {
  color: #2d3748;
  margin-bottom: 20px;
  border-bottom: 3px solid #667eea;
  padding-bottom: 10px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  color: #4a5568;
  font-weight: 600;
  margin-bottom: 8px;
}

.form-group input[type="text"],
.form-group input[type="number"],
.form-group textarea {
  width: 100%;
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1em;
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
}

.form-group input[type="range"] {
  width: 100%;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.submit-btn {
  width: 100%;
  padding: 15px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1em;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-box {
  background: #fed7d7;
  border-left: 4px solid #f56565;
  padding: 15px;
  border-radius: 8px;
  color: #742a2a;
}

.response-box {
  background: #f7fafc;
  border-radius: 8px;
  overflow: hidden;
}

.response-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background: #edf2f7;
  border-bottom: 2px solid #e2e8f0;
}

.badge {
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.9em;
  font-weight: 600;
}

.badge.cached {
  background: #c6f6d5;
  color: #22543d;
}

.badge.fresh {
  background: #fed7d7;
  color: #742a2a;
}

.latency {
  font-weight: 600;
  color: #667eea;
}

.results {
  padding: 15px;
  max-height: 500px;
  overflow-y: auto;
}

.result-item {
  background: white;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
  border-left: 4px solid #667eea;
}

.result-header {
  font-weight: 600;
  color: #667eea;
  margin-bottom: 10px;
}

.result-text {
  color: #2d3748;
  line-height: 1.6;
  white-space: pre-wrap;
}

.response-footer {
  padding: 15px;
  background: #edf2f7;
  border-top: 2px solid #e2e8f0;
  color: #718096;
}

.placeholder {
  padding: 40px;
  text-align: center;
  color: #718096;
}

.placeholder p {
  font-size: 1.2em;
  margin-bottom: 20px;
}

.placeholder ul {
  list-style: none;
  text-align: left;
  display: inline-block;
}

.placeholder li {
  padding: 8px 0;
}

.placeholder li:before {
  content: "✓ ";
  color: #48bb78;
  font-weight: bold;
  margin-right: 8px;
}

.footer {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  text-align: center;
  color: #718096;
}
EOF

# ============================================================================
# Scripts
# ============================================================================

cat > scripts/build.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Building VAIA L2 ==="

# Backend setup (reuse L1's venv concept)
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ../frontend
echo "Installing Node dependencies..."
npm install

echo "✅ Build complete!"
EOF

cat > scripts/start.sh << 'EOF'
#!/bin/bash

echo "=== Starting VAIA L2 ==="

# Start backend
cd backend
source venv/bin/activate
python app/main.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Start frontend
cd ../frontend
PORT=3000 npm start &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo "✅ System running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Docs:     http://localhost:8000/docs"

# Wait for processes
wait
EOF

cat > scripts/stop.sh << 'EOF'
#!/bin/bash

echo "=== Stopping VAIA L2 ==="

# Kill processes on ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "✅ Stopped"
EOF

cat > scripts/test.sh << 'EOF'
#!/bin/bash

API_URL="http://localhost:8000"

case "$1" in
  health)
    echo "=== Health Check ==="
    curl -s "$API_URL/health" | python3 -m json.tool
    ;;
  
  process)
    echo "=== Test Async Processing ==="
    curl -s -X POST "$API_URL/process" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_id": "agent-test1234",
        "prompts": ["What is async/await?", "Explain decorators"],
        "temperature": 0.7,
        "max_tokens": 1024
      }' | python3 -m json.tool
    ;;
  
  metrics)
    echo "=== System Metrics ==="
    curl -s "$API_URL/metrics" | python3 -m json.tool
    ;;
  
  validation)
    echo "=== Test Pydantic Validation ==="
    echo "Sending invalid agent_id..."
    curl -s -X POST "$API_URL/process" \
      -H "Content-Type: application/json" \
      -d '{
        "agent_id": "invalid-id",
        "prompts": ["test"],
        "temperature": 0.7,
        "max_tokens": 1024
      }' | python3 -m json.tool
    ;;
  
  load_test)
    echo "=== Load Test (10 concurrent requests) ==="
    for i in {1..10}; do
      curl -s -X POST "$API_URL/process" \
        -H "Content-Type: application/json" \
        -d "{
          \"agent_id\": \"agent-load${i}0\",
          \"prompts\": [\"Test $i\"],
          \"temperature\": 0.7,
          \"max_tokens\": 100
        }" > /dev/null &
    done
    wait
    echo "Load test complete. Check metrics at $API_URL/metrics"
    ;;
  
  *)
    echo "Usage: $0 {health|process|metrics|validation|load_test}"
    exit 1
    ;;
esac
EOF

chmod +x scripts/*.sh

# ============================================================================
# Documentation
# ============================================================================

cat > README.md << 'EOF'
# VAIA L2: Advanced Python & Libraries

Enterprise-grade async patterns, decorators, and type-safe validation for VAIA systems.

## Quick Start
```bash
# Build (reuses L1 venv patterns)
./scripts/build.sh

# Start system
./scripts/start.sh

# Test endpoints
./scripts/test.sh health
./scripts/test.sh process
./scripts/test.sh metrics
./scripts/test.sh validation
./scripts/test.sh load_test
```

## Features

- **Async Processing**: Non-blocking I/O for 1000+ concurrent requests
- **Production Decorators**: Retry, caching, performance tracking
- **Pydantic Validation**: Type-safe data contracts
- **Real-Time Metrics**: Performance monitoring dashboard

## Architecture

- Backend: Python FastAPI (async/await)
- Frontend: React dashboard
- AI: Gemini API integration
- Patterns: Decorators, async, type safety

## Endpoints

- POST `/process` - Async agent request processing
- GET `/metrics` - System performance metrics
- GET `/health` - Health check
- GET `/docs` - API documentation

## Learning Objectives

1. Master async/await for concurrent I/O
2. Implement production decorators (retry, cache, monitoring)
3. Use Pydantic for data validation
4. Build type-safe VAIA systems

Builds on: L1 Python Setup
Prepares for: L3 Transformer Architecture
EOF

echo ""
echo "=== ✅ VAIA L2 Setup Complete ==="
echo ""
echo "Project structure created in: ./vaia-l2-advanced-python"
echo ""
echo "Next steps:"
echo "  cd vaia-l2-advanced-python"
echo "  ./scripts/build.sh"
echo "  ./scripts/start.sh"
echo ""
echo "Then visit: http://localhost:3000"
echo ""

## DELIVERABLE 3: SVG Diagrams