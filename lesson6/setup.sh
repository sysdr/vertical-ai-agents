#!/bin/bash

# L6: Interacting with LLM APIs - Production-Grade Setup Script
# Builds on L5 model selection framework
# Prepares foundation for L7 prompt engineering

set -e

ROOT_DIR="$(pwd)"

GET_PIP_PATH="/tmp/get-pip.py"

download_get_pip() {
    rm -f "$GET_PIP_PATH"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSLo "$GET_PIP_PATH" https://bootstrap.pypa.io/get-pip.py
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$GET_PIP_PATH" https://bootstrap.pypa.io/get-pip.py
    else
        echo "Need curl or wget to download pip bootstrapper." >&2
        exit 1
    fi
}

ensure_python_tooling() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required. Please install it before running this script." >&2
        exit 1
    fi
    if ! python3 -m venv -h >/dev/null 2>&1; then
        echo "python3-venv is required. Please install it (e.g., apt install python3-venv)." >&2
        exit 1
    fi
}

echo "=================================================="
echo "L6: Interacting with LLM APIs - Setup"
echo "=================================================="

PROJECT_NAME="l6-llm-api-client"
GEMINI_API_KEY="${GEMINI_API_KEY:-mock-api-key}"
USE_MOCK_GEMINI="${USE_MOCK_GEMINI:-true}"

# Ensure Python has venv/pip available
echo "[0/10] Verifying Python tooling..."
ensure_python_tooling

# Create project structure
echo "[1/10] Creating project structure..."
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,tests,config,logs}
cd $PROJECT_NAME

# Create Python virtual environment
echo "[2/10] Setting up Python environment..."
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv --without-pip venv
source venv/bin/activate

if ! command -v pip >/dev/null 2>&1; then
    echo "[2/10] Bootstrapping pip inside virtualenv..."
    download_get_pip
    python "$GET_PIP_PATH"
fi

# Install dependencies
echo "[3/10] Installing Python dependencies..."
cat > requirements.txt << 'EOF'
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1
pydantic-settings==2.1.0
aiohttp==3.9.3
python-dotenv==1.0.1
google-generativeai==0.4.0
redis==5.0.1
prometheus-client==0.19.0
structlog==24.1.0
pytest==7.4.4
pytest-asyncio==0.23.4
httpx==0.26.0
EOF

pip install --upgrade pip
pip install -r requirements.txt

# Create environment configuration
echo "[4/10] Creating environment configuration..."
cat > .env << EOF
# Gemini API Configuration
GEMINI_API_KEY=$GEMINI_API_KEY
GEMINI_MODEL=gemini-1.5-flash
USE_MOCK_GEMINI=$USE_MOCK_GEMINI

# Rate Limiting (per minute)
RATE_LIMIT_RPM=60
RATE_LIMIT_TPM=100000

# Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60

# Cost Configuration (USD per 1M tokens)
COST_INPUT_TOKEN=0.075
COST_OUTPUT_TOKEN=0.300

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ENABLED=false

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
EOF

# Create configuration module
echo "[5/10] Creating configuration module..."
cat > config/settings.py << 'EOF'
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    gemini_api_key: str = "mock-api-key"
    gemini_model: str = "gemini-1.5-flash"
    use_mock_gemini: bool = True
    
    # Rate Limiting
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    
    # Circuit Breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Cost Configuration
    cost_input_token: float = 0.075  # per 1M tokens
    cost_output_token: float = 0.300  # per 1M tokens
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_enabled: bool = False
    
    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
EOF

# Create rate limiter module
echo "[6/10] Creating rate limiter..."
cat > backend/rate_limiter.py << 'EOF'
import time
import asyncio
from typing import Dict
import structlog

logger = structlog.get_logger()

class RateLimiter:
    """Token bucket algorithm for RPM and TPM rate limiting."""
    
    def __init__(self, rpm: int, tpm: int):
        self.rpm_capacity = rpm
        self.tpm_capacity = tpm
        self.rpm_tokens = float(rpm)
        self.tpm_tokens = float(tpm)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
        logger.info("rate_limiter_initialized", 
                   rpm_capacity=rpm, tpm_capacity=tpm)
    
    async def acquire(self, token_count: int = 1) -> Dict[str, any]:
        """
        Attempt to acquire tokens for a request.
        Returns dict with success status and wait time if throttled.
        """
        async with self.lock:
            self._refill()
            
            if self.rpm_tokens >= 1 and self.tpm_tokens >= token_count:
                self.rpm_tokens -= 1
                self.tpm_tokens -= token_count
                
                logger.debug("rate_limit_acquired",
                           rpm_remaining=int(self.rpm_tokens),
                           tpm_remaining=int(self.tpm_tokens))
                
                return {
                    "allowed": True,
                    "rpm_remaining": int(self.rpm_tokens),
                    "tpm_remaining": int(self.tpm_tokens)
                }
            else:
                # Calculate wait time for token replenishment
                wait_time = self._calculate_wait_time(token_count)
                
                logger.warning("rate_limit_exceeded",
                             rpm_remaining=int(self.rpm_tokens),
                             tpm_remaining=int(self.tpm_tokens),
                             wait_seconds=wait_time)
                
                return {
                    "allowed": False,
                    "wait_seconds": wait_time,
                    "rpm_remaining": int(self.rpm_tokens),
                    "tpm_remaining": int(self.tpm_tokens)
                }
    
    def _refill(self):
        """Continuously refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Refill RPM tokens
        rpm_refill = (self.rpm_capacity / 60.0) * elapsed
        self.rpm_tokens = min(self.rpm_capacity, 
                             self.rpm_tokens + rpm_refill)
        
        # Refill TPM tokens
        tpm_refill = (self.tpm_capacity / 60.0) * elapsed
        self.tpm_tokens = min(self.tpm_capacity,
                             self.tpm_tokens + tpm_refill)
        
        self.last_refill = now
    
    def _calculate_wait_time(self, token_count: int) -> float:
        """Calculate seconds to wait for token replenishment."""
        if self.rpm_tokens < 1:
            rpm_wait = (1 - self.rpm_tokens) / (self.rpm_capacity / 60.0)
        else:
            rpm_wait = 0
        
        if self.tpm_tokens < token_count:
            tpm_wait = (token_count - self.tpm_tokens) / (self.tpm_capacity / 60.0)
        else:
            tpm_wait = 0
        
        return max(rpm_wait, tpm_wait)
    
    def get_stats(self) -> Dict[str, any]:
        """Get current rate limiter statistics."""
        return {
            "rpm_available": int(self.rpm_tokens),
            "rpm_capacity": self.rpm_capacity,
            "tpm_available": int(self.tpm_tokens),
            "tpm_capacity": self.tpm_capacity,
            "utilization_rpm": 1 - (self.rpm_tokens / self.rpm_capacity),
            "utilization_tpm": 1 - (self.tpm_tokens / self.tpm_capacity)
        }
EOF

# Create circuit breaker module
cat > backend/circuit_breaker.py << 'EOF'
import time
import asyncio
from enum import Enum
import structlog

logger = structlog.get_logger()

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = asyncio.Lock()
        
        logger.info("circuit_breaker_initialized",
                   failure_threshold=failure_threshold,
                   timeout=timeout)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    logger.info("circuit_breaker_half_open",
                              failures=self.failure_count)
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open. Retry after {self.timeout}s"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful call."""
        async with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:  # Require 2 successes to close
                    logger.info("circuit_breaker_closed",
                              successes=self.success_count)
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    async def _record_failure(self):
        """Record failed call."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                logger.error("circuit_breaker_opened",
                           failures=self.failure_count,
                           timeout=self.timeout)
                self.state = CircuitState.OPEN
                self.success_count = 0
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "threshold": self.failure_threshold
        }
EOF

# Create cost tracker module
cat > backend/cost_tracker.py << 'EOF'
from typing import Dict
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class CostTracker:
    """Track API usage and costs in real-time."""
    
    def __init__(self, input_cost_per_1m: float, output_cost_per_1m: float):
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        
        # Usage counters
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        # Session tracking
        self.session_start = datetime.now()
        
        logger.info("cost_tracker_initialized",
                   input_cost=input_cost_per_1m,
                   output_cost=output_cost_per_1m)
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage and calculate cost."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        request_cost = input_cost + output_cost
        
        self.total_cost += request_cost
        
        logger.info("usage_recorded",
                   input_tokens=input_tokens,
                   output_tokens=output_tokens,
                   cost_usd=round(request_cost, 6))
        
        return request_cost
    
    def get_stats(self) -> Dict[str, any]:
        """Get usage statistics."""
        uptime = datetime.now() - self.session_start
        avg_cost_per_request = (self.total_cost / self.total_requests 
                                if self.total_requests > 0 else 0)
        
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_request": round(avg_cost_per_request, 6),
            "uptime_seconds": int(uptime.total_seconds()),
            "requests_per_hour": int(self.total_requests / (uptime.total_seconds() / 3600))
                if uptime.total_seconds() > 0 else 0
        }
    
    def reset_stats(self):
        """Reset all counters."""
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.session_start = datetime.now()
        
        logger.info("cost_tracker_reset")
EOF

# Create Gemini API client
echo "[7/10] Creating Gemini API client..."
cat > backend/gemini_client.py << 'EOF'
import google.generativeai as genai
import asyncio
import time
import random
from typing import Dict
import structlog
from config.settings import settings
from backend.rate_limiter import RateLimiter
from backend.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from backend.cost_tracker import CostTracker

logger = structlog.get_logger()

class GeminiAPIClient:
    """Production-grade Gemini API client with rate limiting, circuit breaker, cost tracking, and mock mode."""
    
    def __init__(self):
        self.use_mock = settings.use_mock_gemini

        # Configure Gemini when not in mock mode
        if not self.use_mock:
            genai.configure(api_key=settings.gemini_api_key)
            self.model = genai.GenerativeModel(settings.gemini_model)
        else:
            self.model = None
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            rpm=settings.rate_limit_rpm,
            tpm=settings.rate_limit_tpm
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout
        )
        self.cost_tracker = CostTracker(
            input_cost_per_1m=settings.cost_input_token,
            output_cost_per_1m=settings.cost_output_token
        )
        
        logger.info("gemini_client_initialized", model=settings.gemini_model, use_mock=self.use_mock)
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, any]:
        """
        Generate content with full production patterns:
        - Rate limiting
        - Circuit breaker
        - Retry with exponential backoff
        - Cost tracking
        - Structured logging
        - Optional mock mode for offline/demo use
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        logger.info("request_started", request_id=request_id, prompt_length=len(prompt), use_mock=self.use_mock)
        
        try:
            # Estimate tokens (rough approximation: 4 chars per token)
            estimated_tokens = max(1, len(prompt) // 4)
            
            # Rate limiting
            rate_limit_result = await self.rate_limiter.acquire(estimated_tokens)
            if not rate_limit_result["allowed"]:
                logger.warning("request_throttled", 
                             request_id=request_id,
                             wait_seconds=rate_limit_result["wait_seconds"])
                return {
                    "success": False,
                    "error": "rate_limited",
                    "wait_seconds": rate_limit_result["wait_seconds"],
                    "request_id": request_id
                }
            
            # Execute generation (mock or real)
            if self.use_mock:
                response = await self._mock_generate_response(prompt, **kwargs)
            else:
                response = await self.circuit_breaker.call(
                    self._generate_with_retry,
                    prompt,
                    **kwargs
                )
            
            # Track cost
            input_tokens = response.get("input_tokens", estimated_tokens)
            output_tokens = response.get("output_tokens", max(1, len(response.get("text", "")) // 4))
            cost = self.cost_tracker.record_usage(input_tokens, output_tokens)
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("request_completed",
                       request_id=request_id,
                       latency_ms=latency_ms,
                       cost_usd=round(cost, 6),
                       use_mock=self.use_mock)
            
            return {
                "success": True,
                "text": response["text"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
                "latency_ms": latency_ms,
                "request_id": request_id
            }
            
        except CircuitBreakerOpenError as e:
            logger.error("circuit_breaker_blocked", request_id=request_id)
            return {
                "success": False,
                "error": "circuit_breaker_open",
                "message": str(e),
                "request_id": request_id
            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("request_failed",
                        request_id=request_id,
                        error=str(e),
                        latency_ms=latency_ms)
            return {
                "success": False,
                "error": "api_error",
                "message": str(e),
                "request_id": request_id
            }
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> Dict:
        """Generate with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                # Run synchronous API call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(prompt, **kwargs)
                )
                
                return {
                    "text": response.text,
                    "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0)
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                base_delay = 0.1 * (2 ** attempt)
                jitter = base_delay * random.random()
                wait_time = base_delay + jitter
                
                logger.warning("retry_attempt",
                             attempt=attempt + 1,
                             max_retries=max_retries,
                             wait_seconds=round(wait_time, 2),
                             error=str(e))
                
                await asyncio.sleep(wait_time)
    
    async def _mock_generate_response(self, prompt: str, **kwargs) -> Dict[str, any]:
        """Mock response for offline/testing/demo runs."""
        await asyncio.sleep(0.05 + random.random() * 0.05)
        output_tokens = max(24, min(256, len(prompt) // 3 + 40))
        input_tokens = max(16, len(prompt) // 4 or 16)
        mock_text = f"[MOCK] Generated response for: {prompt[:80] or 'empty prompt'}"
        
        logger.debug("mock_response_generated",
                     output_tokens=output_tokens,
                     input_tokens=input_tokens)
        
        return {
            "text": mock_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive client statistics."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "cost_tracker": self.cost_tracker.get_stats()
        }
EOF

# Create FastAPI backend
cat > backend/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import structlog
from backend.gemini_client import GeminiAPIClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

app = FastAPI(title="L6: LLM API Client", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize client
client = GeminiAPIClient()

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class GenerateResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    request_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    wait_seconds: Optional[float] = None

@app.get("/")
async def root():
    return {
        "service": "L6: LLM API Client",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate content with production-grade API client."""
    result = await client.generate(
        prompt=request.prompt,
        generation_config={
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens
        }
    )
    return GenerateResponse(**result)

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive client statistics."""
    return client.get_stats()

@app.post("/api/stats/reset")
async def reset_stats():
    """Reset cost tracker statistics."""
    client.cost_tracker.reset_stats()
    return {"message": "Statistics reset successfully"}
EOF

# Create React frontend
echo "[8/10] Creating React frontend..."
cat > frontend/package.json << 'EOF'
{
  "name": "l6-llm-api-client-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "recharts": "^2.10.4",
    "axios": "^1.6.7"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": ["react-app"]
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>L6: LLM API Client</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/stats`);
      setStats(res.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await axios.post(`${API_BASE}/api/generate`, {
        prompt: prompt,
        temperature: 0.7,
        max_tokens: 1000
      });

      setResponse(res.data);
      
      if (res.data.success) {
        setHistory(prev => [{
          prompt: prompt.substring(0, 50) + '...',
          tokens: res.data.output_tokens,
          cost: res.data.cost_usd,
          latency: res.data.latency_ms,
          timestamp: new Date().toISOString()
        }, ...prev].slice(0, 10));
      }
    } catch (error) {
      setResponse({
        success: false,
        error: 'network_error',
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_BASE}/api/stats/reset`);
      fetchStats();
      setHistory([]);
    } catch (error) {
      console.error('Error resetting stats:', error);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🚀 L6: LLM API Client</h1>
        <p>Production-Grade API Integration with Rate Limiting & Circuit Breaker</p>
      </header>

      <div className="container">
        {/* Stats Dashboard */}
        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <h3>Rate Limiter</h3>
              <div className="stat-value">{stats.rate_limiter.rpm_available}</div>
              <div className="stat-label">RPM Available</div>
              <div className="stat-subvalue">
                {Math.round(stats.rate_limiter.utilization_rpm * 100)}% utilized
              </div>
            </div>

            <div className="stat-card">
              <h3>Circuit Breaker</h3>
              <div className={`stat-value status-${stats.circuit_breaker.state}`}>
                {stats.circuit_breaker.state.toUpperCase()}
              </div>
              <div className="stat-label">Status</div>
              <div className="stat-subvalue">
                {stats.circuit_breaker.failure_count} failures
              </div>
            </div>

            <div className="stat-card">
              <h3>Cost Tracking</h3>
              <div className="stat-value">${stats.cost_tracker.total_cost_usd}</div>
              <div className="stat-label">Total Cost</div>
              <div className="stat-subvalue">
                {stats.cost_tracker.total_requests} requests
              </div>
            </div>

            <div className="stat-card">
              <h3>Performance</h3>
              <div className="stat-value">
                ${(stats.cost_tracker.avg_cost_per_request * 1000).toFixed(3)}
              </div>
              <div className="stat-label">Avg Cost ($/1K req)</div>
              <div className="stat-subvalue">
                {stats.cost_tracker.requests_per_hour} req/hr
              </div>
            </div>
          </div>
        )}

        {/* Prompt Interface */}
        <div className="prompt-section">
          <h2>Generate Content</h2>
          <textarea
            className="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here... (try: 'Write a haiku about AI')"
            rows={4}
          />
          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
            >
              {loading ? '⏳ Generating...' : '✨ Generate'}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              🔄 Reset Stats
            </button>
          </div>
        </div>

        {/* Response Display */}
        {response && (
          <div className={`response-section ${response.success ? 'success' : 'error'}`}>
            <h2>{response.success ? '✅ Response' : '❌ Error'}</h2>
            
            {response.success ? (
              <>
                <div className="response-text">{response.text}</div>
                <div className="response-meta">
                  <span>📊 {response.input_tokens} → {response.output_tokens} tokens</span>
                  <span>💰 ${response.cost_usd.toFixed(6)}</span>
                  <span>⚡ {response.latency_ms}ms</span>
                  <span>🔖 {response.request_id}</span>
                </div>
              </>
            ) : (
              <div className="error-message">
                <strong>{response.error}:</strong> {response.message}
                {response.wait_seconds && (
                  <p>⏰ Rate limited. Retry in {response.wait_seconds.toFixed(1)}s</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Request History */}
        {history.length > 0 && (
          <div className="history-section">
            <h2>📜 Recent Requests</h2>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                  <th>Latency</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item, idx) => (
                  <tr key={idx}>
                    <td>{item.prompt}</td>
                    <td>{item.tokens}</td>
                    <td>${item.cost.toFixed(6)}</td>
                    <td>{item.latency}ms</td>
                    <td>{new Date(item.timestamp).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>L6: Interacting with LLM APIs | VAIA Curriculum</p>
        <p>Built with FastAPI + React | Powered by Gemini AI</p>
      </footer>
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
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: white;
  padding: 2rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header h1 {
  color: #667eea;
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.header p {
  color: #666;
  font-size: 1.1rem;
}

.container {
  flex: 1;
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
  width: 100%;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.stat-card h3 {
  color: #667eea;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 1rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: bold;
  color: #333;
  margin-bottom: 0.5rem;
}

.stat-value.status-closed {
  color: #10b981;
}

.stat-value.status-open {
  color: #ef4444;
}

.stat-value.status-half_open {
  color: #f59e0b;
}

.stat-label {
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 0.25rem;
}

.stat-subvalue {
  color: #999;
  font-size: 0.85rem;
}

.prompt-section {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin-bottom: 2rem;
}

.prompt-section h2 {
  color: #667eea;
  margin-bottom: 1rem;
}

.prompt-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.2s;
}

.prompt-input:focus {
  outline: none;
  border-color: #667eea;
}

.button-group {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.btn {
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn-secondary {
  background: #f3f4f6;
  color: #374151;
}

.btn-secondary:hover:not(:disabled) {
  background: #e5e7eb;
}

.response-section {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin-bottom: 2rem;
}

.response-section.success {
  border-left: 4px solid #10b981;
}

.response-section.error {
  border-left: 4px solid #ef4444;
}

.response-section h2 {
  margin-bottom: 1rem;
}

.response-text {
  background: #f9fafb;
  padding: 1.5rem;
  border-radius: 8px;
  line-height: 1.6;
  margin-bottom: 1rem;
  white-space: pre-wrap;
}

.response-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  color: #666;
  font-size: 0.9rem;
}

.error-message {
  background: #fef2f2;
  padding: 1rem;
  border-radius: 8px;
  color: #991b1b;
}

.history-section {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.history-section h2 {
  color: #667eea;
  margin-bottom: 1rem;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
}

.history-table th,
.history-table td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e5e7eb;
}

.history-table th {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
}

.history-table tr:hover {
  background: #f9fafb;
}

.footer {
  background: rgba(255,255,255,0.1);
  color: white;
  padding: 2rem;
  text-align: center;
  margin-top: auto;
}

.footer p {
  margin: 0.25rem 0;
}

@media (max-width: 768px) {
  .header h1 {
    font-size: 1.75rem;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .button-group {
    flex-direction: column;
  }

  .response-meta {
    flex-direction: column;
    gap: 0.5rem;
  }
}
EOF

cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

# Create test suite
echo "[9/10] Creating test suite..."
cat > tests/test_api_client.py << 'EOF'
import pytest
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure tests run in mock mode and never hit external APIs
os.environ.setdefault("USE_MOCK_GEMINI", "true")
os.environ.setdefault("GEMINI_API_KEY", "mock-api-key")

from backend.gemini_client import GeminiAPIClient
from backend.rate_limiter import RateLimiter
from backend.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from backend.cost_tracker import CostTracker

@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test rate limiter allows requests within limits."""
    limiter = RateLimiter(rpm=60, tpm=10000)
    
    # Should allow first request
    result = await limiter.acquire(100)
    assert result["allowed"] == True
    assert result["rpm_remaining"] == 59

@pytest.mark.asyncio
async def test_rate_limiter_throttling():
    """Test rate limiter throttles when limits exceeded."""
    limiter = RateLimiter(rpm=2, tpm=100)
    
    # Use up all tokens
    await limiter.acquire(50)
    await limiter.acquire(50)
    
    # Next request should be throttled
    result = await limiter.acquire(10)
    assert result["allowed"] == False
    assert "wait_seconds" in result

@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1)
    
    async def failing_func():
        raise Exception("Simulated failure")
    
    # Trigger failures
    for _ in range(3):
        try:
            await breaker.call(failing_func)
        except:
            pass
    
    # Circuit should be open now
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(failing_func)

@pytest.mark.asyncio
async def test_cost_tracker():
    """Test cost tracking accuracy."""
    tracker = CostTracker(input_cost_per_1m=0.075, output_cost_per_1m=0.300)
    
    # Record usage
    cost = tracker.record_usage(input_tokens=1000, output_tokens=500)
    
    # Verify calculation
    expected_cost = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.300
    assert abs(cost - expected_cost) < 0.000001
    
    stats = tracker.get_stats()
    assert stats["total_requests"] == 1
    assert stats["total_input_tokens"] == 1000

@pytest.mark.asyncio
async def test_gemini_client_generate():
    """Test Gemini client generates content successfully."""
    client = GeminiAPIClient()
    
    result = await client.generate("Say hello in one word")
    
    assert result["success"] == True
    assert "text" in result
    assert result["cost_usd"] > 0
    assert result["latency_ms"] > 0

@pytest.mark.asyncio
async def test_gemini_client_stats():
    """Test client statistics collection."""
    client = GeminiAPIClient()
    
    await client.generate("Test prompt")
    
    stats = client.get_stats()
    assert "rate_limiter" in stats
    assert "circuit_breaker" in stats
    assert "cost_tracker" in stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create build, start, stop, test scripts
cat > build.sh << 'EOF'
#!/bin/bash
set -e

echo "Building L6: LLM API Client..."

# Backend
echo "[1/2] Setting up Python environment..."
source venv/bin/activate
pip install -r requirements.txt

# Frontend
echo "[2/2] Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo "Run './start.sh' to start the application"
EOF

cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting L6: LLM API Client..."

# Start backend
echo "[1/2] Starting FastAPI backend..."
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
echo "[2/2] Starting React frontend..."
cd frontend
BROWSER=none npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Application started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
EOF

cat > stop.sh << 'EOF'
#!/bin/bash

echo "Stopping L6: LLM API Client..."

# Kill backend
pkill -f "uvicorn backend.main:app" || true

# Kill frontend
pkill -f "react-scripts start" || true

echo "✅ All services stopped"
EOF

cat > test.sh << 'EOF'
#!/bin/bash
set -e

echo "Running L6 Tests..."
source venv/bin/activate

if [ "$1" == "rate_limit" ]; then
    echo "Testing rate limiter..."
    python -m pytest tests/test_api_client.py::test_rate_limiter_basic -v
    python -m pytest tests/test_api_client.py::test_rate_limiter_throttling -v
elif [ "$1" == "circuit_breaker" ]; then
    echo "Testing circuit breaker..."
    python -m pytest tests/test_api_client.py::test_circuit_breaker_opens -v
elif [ "$1" == "cost_tracking" ]; then
    echo "Testing cost tracker..."
    python -m pytest tests/test_api_client.py::test_cost_tracker -v
else
    echo "Running all tests..."
    python -m pytest tests/test_api_client.py -v
fi

echo "✅ Tests complete!"
EOF

chmod +x build.sh start.sh stop.sh test.sh

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm curl

# Copy backend files
COPY requirements.txt .
COPY backend/ backend/
COPY config/ config/
COPY .env .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend files
COPY frontend/package.json frontend/
WORKDIR /app/frontend
RUN npm install

COPY frontend/ .
RUN npm run build

# Return to app directory
WORKDIR /app

# Expose ports
EXPOSE 8000 3000

# Start script
COPY docker-start.sh .
RUN chmod +x docker-start.sh

CMD ["./docker-start.sh"]
EOF

cat > docker-start.sh << 'EOF'
#!/bin/bash

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Serve frontend build
cd frontend
npx serve -s build -l 3000 &

wait
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  l6-api-client:
    build: .
    ports:
      - "8000:8000"
      - "3000:3000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./logs:/app/logs
EOF

# Create README
cat > README.md << 'EOF'
# L6: Interacting with LLM APIs - Production-Grade Integration

## Quick Start

### Option 1: Local Development
```bash
# Build
./build.sh

# Start
./start.sh

# Access
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs

# Stop
./stop.sh

# Test
./test.sh
```

### Option 2: Docker
```bash
docker-compose up --build
```

## Features

- ✅ Token bucket rate limiting (RPM + TPM)
- ✅ Circuit breaker pattern for fault tolerance
- ✅ Exponential backoff with jitter
- ✅ Real-time cost tracking
- ✅ Structured logging
- ✅ Production-grade async I/O
- ✅ Comprehensive testing

## Architecture

```
React UI (3000) ──→ FastAPI (8000) ──→ Gemini AI
                         │
                         ├─ Rate Limiter
                         ├─ Circuit Breaker
                         └─ Cost Tracker
```

## Testing

```bash
# All tests
./test.sh

# Specific tests
./test.sh rate_limit
./test.sh circuit_breaker
./test.sh cost_tracking
```

## Configuration

Edit `.env` to customize:
- Rate limits (RPM/TPM)
- Circuit breaker thresholds
- Cost parameters
- Model selection

## Next Steps

This foundation prepares you for:
- L7: Basic LLM Prompting & JSON Output
- L8: Streaming Responses
- L15: Multi-Model Orchestration
EOF

echo "[10/10] Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ L6: LLM API Client - Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Project structure created in: ./$PROJECT_NAME"
echo ""
echo "🚀 Next steps:"
echo "   cd $PROJECT_NAME"
echo "   ./build.sh    # Install dependencies"
echo "   ./start.sh    # Start application"
echo "   ./test.sh     # Run tests"
echo ""
echo "🎯 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"