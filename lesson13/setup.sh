#!/bin/bash

# L13: Context Engineering - Automated Setup Script
# Builds on L12 few-shot components, enables L14 state management

set -e

PROJECT_NAME="vaia-l13-context-engineering"
GEMINI_API_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"

echo "=================================="
echo "VAIA L13: Context Engineering Setup"
echo "=================================="

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend,docker,docs}
cd $PROJECT_NAME

# Backend structure
mkdir -p backend/{api,services,utils,models,tests}
mkdir -p backend/api/{routes,middleware}

# Frontend structure  
mkdir -p frontend/{src,public}
mkdir -p frontend/src/{components,services,hooks,utils}

echo "✓ Project structure created"

# ==================== BACKEND FILES ====================

# Backend requirements
cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
google-generativeai==0.3.2
tiktoken==0.5.2
nltk==3.8.1
scikit-learn==1.4.0
numpy==1.26.3
aiohttp==3.9.1
prometheus-client==0.19.0
pytest==7.4.4
pytest-asyncio==0.23.3
httpx==0.26.0
EOF

# Backend .env
cat > backend/.env << EOF
GEMINI_API_KEY=$GEMINI_API_KEY
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_CONTEXT_TOKENS=30000
COMPRESSION_TIMEOUT_MS=500
EOF

# Main FastAPI application
cat > backend/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from api.routes import context_routes
from utils.token_counter import TokenCounter
from services.summarizer import SummarizerService
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Download NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")
    
    yield
    
    logger.info("Shutting down Context Engineering service")

app = FastAPI(
    title="VAIA L13: Context Engineering",
    description="Production-grade context management and optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(context_routes.router, prefix="/api/v1", tags=["context"])

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "context-engineering",
        "lesson": "L13"
    }

@app.get("/")
async def root():
    return {
        "message": "VAIA L13: Context Engineering API",
        "endpoints": [
            "/api/v1/count-tokens",
            "/api/v1/summarize",
            "/api/v1/optimize-context",
            "/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Token Counter utility
cat > backend/utils/token_counter.py << 'EOF'
import tiktoken
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    """Production-grade token counting using tiktoken"""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize with model-specific encoding"""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self.model = model
            logger.info(f"TokenCounter initialized for model: {model}")
        except KeyError:
            # Fallback to cl100k_base (GPT-4, Gemini compatible)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = "cl100k_base"
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting error: {e}")
            # Rough fallback: 4 chars per token average
            return len(text) // 4
    
    def count_messages(self, messages: list) -> int:
        """Count tokens in message array (chat format)"""
        total = 0
        for message in messages:
            # Add message formatting overhead (role, delimiters)
            total += 4  # Overhead per message
            
            if isinstance(message, dict):
                for key, value in message.items():
                    total += self.count(str(value))
            else:
                total += self.count(str(message))
        
        total += 2  # Overall conversation overhead
        return total
    
    def analyze(self, text: str, max_tokens: int = 30000) -> Dict:
        """Analyze token usage with recommendations"""
        token_count = self.count(text)
        usage_percent = (token_count / max_tokens) * 100
        
        # Determine recommendation
        if usage_percent < 70:
            recommendation = "optimal"
            action = "none"
        elif usage_percent < 90:
            recommendation = "moderate"
            action = "light_compression"
        else:
            recommendation = "critical"
            action = "aggressive_compression"
        
        return {
            "token_count": token_count,
            "max_tokens": max_tokens,
            "usage_percent": round(usage_percent, 2),
            "remaining_tokens": max_tokens - token_count,
            "recommendation": recommendation,
            "action": action,
            "model": self.model
        }
    
    def estimate_cost(self, token_count: int, price_per_1k: float = 0.01) -> float:
        """Estimate API cost based on token count"""
        return (token_count / 1000) * price_per_1k

# Global instance
token_counter = TokenCounter()
EOF

# Summarizer service
cat > backend/services/summarizer.py << 'EOF'
import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import os
import asyncio
from functools import lru_cache
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class SummarizerService:
    """Multi-strategy text summarization for context compression"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        logger.info("SummarizerService initialized")
    
    async def extractive_summarize(
        self, 
        text: str, 
        ratio: float = 0.3,
        min_sentences: int = 2
    ) -> str:
        """
        Extractive summarization: select most important sentences
        Fast, preserves exact wording
        """
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= min_sentences:
                return text
            
            # Calculate sentence scores using TF-IDF
            scores = self._score_sentences(sentences)
            
            # Select top sentences
            target_count = max(min_sentences, int(len(sentences) * ratio))
            top_indices = np.argsort(scores)[-target_count:]
            
            # Maintain original order
            selected_sentences = [sentences[i] for i in sorted(top_indices)]
            
            summary = ' '.join(selected_sentences)
            logger.info(f"Extractive: {len(sentences)} → {len(selected_sentences)} sentences")
            
            return summary
            
        except Exception as e:
            logger.error(f"Extractive summarization error: {e}")
            return text[:len(text)//2]  # Fallback: truncate
    
    def _score_sentences(self, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF and position"""
        if len(sentences) == 0:
            return np.array([])
        
        # TF-IDF scoring
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        except:
            # Fallback: simple word count
            sentence_scores = np.array([len(s.split()) for s in sentences])
        
        # Position bonus (first and last sentences often important)
        position_weights = np.ones(len(sentences))
        position_weights[0] = 1.5  # First sentence bonus
        position_weights[-1] = 1.3  # Last sentence bonus
        
        return sentence_scores * position_weights
    
    async def abstractive_summarize(
        self, 
        text: str, 
        target_ratio: float = 0.5,
        timeout: float = 10.0
    ) -> str:
        """
        Abstractive summarization using Gemini
        Slower, more compression, may lose nuance
        """
        try:
            # Calculate target length
            current_length = len(text)
            target_length = int(current_length * target_ratio)
            
            prompt = f"""Summarize the following text concisely, reducing it to approximately {target_length} characters while preserving all key information and technical details.

Text to summarize:
{text}

Concise summary:"""
            
            # Generate with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config={'temperature': 0.3, 'max_output_tokens': 2048}
                ),
                timeout=timeout
            )
            
            summary = response.text.strip()
            
            logger.info(f"Abstractive: {current_length} → {len(summary)} chars")
            return summary
            
        except asyncio.TimeoutError:
            logger.warning("Abstractive summarization timeout, using extractive fallback")
            return await self.extractive_summarize(text, ratio=target_ratio)
        except Exception as e:
            logger.error(f"Abstractive summarization error: {e}")
            return await self.extractive_summarize(text, ratio=target_ratio)
    
    async def hybrid_summarize(
        self, 
        text: str, 
        ratio: float = 0.4
    ) -> str:
        """
        Hybrid: extractive selection then abstractive compression
        Balanced speed and quality
        """
        try:
            # Step 1: Extractive selection (get to 60% of target)
            extractive_ratio = ratio * 1.5
            extracted = await self.extractive_summarize(text, ratio=extractive_ratio)
            
            # Step 2: Abstractive compression of extracted content
            if len(extracted) < len(text) * 0.8:  # Only if extraction helped
                summary = await self.abstractive_summarize(extracted, target_ratio=0.8)
                logger.info(f"Hybrid: {len(text)} → {len(summary)} chars")
                return summary
            else:
                return extracted
                
        except Exception as e:
            logger.error(f"Hybrid summarization error: {e}")
            return await self.extractive_summarize(text, ratio=ratio)
    
    async def summarize(
        self, 
        text: str, 
        strategy: str = "extractive",
        target_ratio: float = 0.3
    ) -> Dict:
        """
        Main summarization interface
        Returns summary with metadata
        """
        import time
        start_time = time.time()
        
        original_length = len(text)
        
        # Select strategy
        if strategy == "extractive":
            summary = await self.extractive_summarize(text, ratio=target_ratio)
        elif strategy == "abstractive":
            summary = await self.abstractive_summarize(text, target_ratio=target_ratio)
        elif strategy == "hybrid":
            summary = await self.hybrid_summarize(text, ratio=target_ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        compression_time = time.time() - start_time
        compressed_length = len(summary)
        compression_ratio = original_length / compressed_length if compressed_length > 0 else 1.0
        
        return {
            "original_text": text,
            "summary": summary,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": round(compression_ratio, 2),
            "strategy": strategy,
            "target_ratio": target_ratio,
            "compression_time_ms": round(compression_time * 1000, 2)
        }

# Global instance
summarizer = SummarizerService()
EOF

# Context optimization service
cat > backend/services/context_optimizer.py << 'EOF'
from typing import Dict, List, Optional
import logging
from utils.token_counter import token_counter
from services.summarizer import summarizer
import os

logger = logging.getLogger(__name__)

class ContextOptimizer:
    """Orchestrates full context optimization pipeline"""
    
    def __init__(self):
        self.max_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))
        self.compression_timeout = int(os.getenv("COMPRESSION_TIMEOUT_MS", "500")) / 1000
        logger.info(f"ContextOptimizer initialized (max_tokens={self.max_tokens})")
    
    async def optimize(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        preserve_quality: bool = True
    ) -> Dict:
        """
        Optimize context with intelligent strategy selection
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Step 1: Analyze current state
        analysis = token_counter.analyze(text, max_tokens)
        
        # Step 2: Decide on strategy
        if analysis["recommendation"] == "optimal":
            # No compression needed
            return {
                "optimized_text": text,
                "original_tokens": analysis["token_count"],
                "optimized_tokens": analysis["token_count"],
                "compression_ratio": 1.0,
                "strategy": "none",
                "action_taken": "pass_through",
                "quality_preserved": True,
                "cost_savings": 0
            }
        
        # Step 3: Select compression strategy
        if analysis["recommendation"] == "moderate":
            strategy = "extractive"
            target_ratio = 0.7  # Light compression
        else:  # critical
            strategy = "hybrid"
            target_ratio = 0.5  # Aggressive compression
        
        # Step 4: Compress
        try:
            result = await summarizer.summarize(
                text,
                strategy=strategy,
                target_ratio=target_ratio
            )
            
            # Step 5: Validate compression quality
            compressed_tokens = token_counter.count(result["summary"])
            
            # Check if compression helped
            if compressed_tokens >= analysis["token_count"] * 0.9:
                # Compression didn't help enough, truncate instead
                logger.warning("Compression ineffective, truncating")
                truncated = self._truncate_text(text, max_tokens)
                truncated_tokens = token_counter.count(truncated)
                
                return {
                    "optimized_text": truncated,
                    "original_tokens": analysis["token_count"],
                    "optimized_tokens": truncated_tokens,
                    "compression_ratio": analysis["token_count"] / truncated_tokens,
                    "strategy": "truncation",
                    "action_taken": "compression_failed_truncated",
                    "quality_preserved": False,
                    "cost_savings": self._calculate_savings(
                        analysis["token_count"], 
                        truncated_tokens
                    )
                }
            
            # Compression succeeded
            return {
                "optimized_text": result["summary"],
                "original_tokens": analysis["token_count"],
                "optimized_tokens": compressed_tokens,
                "compression_ratio": result["compression_ratio"],
                "strategy": strategy,
                "action_taken": "compressed",
                "quality_preserved": preserve_quality,
                "compression_time_ms": result["compression_time_ms"],
                "cost_savings": self._calculate_savings(
                    analysis["token_count"],
                    compressed_tokens
                )
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            # Fallback to truncation
            truncated = self._truncate_text(text, max_tokens)
            truncated_tokens = token_counter.count(truncated)
            
            return {
                "optimized_text": truncated,
                "original_tokens": analysis["token_count"],
                "optimized_tokens": truncated_tokens,
                "compression_ratio": analysis["token_count"] / truncated_tokens,
                "strategy": "truncation",
                "action_taken": "error_truncated",
                "quality_preserved": False,
                "error": str(e),
                "cost_savings": self._calculate_savings(
                    analysis["token_count"],
                    truncated_tokens
                )
            }
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Intelligently truncate text to fit token limit"""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        
        if last_period > max_chars * 0.8:  # If period is in last 20%
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _calculate_savings(self, original_tokens: int, optimized_tokens: int) -> Dict:
        """Calculate cost savings from optimization"""
        tokens_saved = original_tokens - optimized_tokens
        cost_per_1k = 0.01  # $0.01 per 1K tokens (example pricing)
        
        money_saved = (tokens_saved / 1000) * cost_per_1k
        
        return {
            "tokens_saved": tokens_saved,
            "percent_reduction": round((tokens_saved / original_tokens) * 100, 2),
            "money_saved_per_request": round(money_saved, 6),
            "projected_monthly_savings": round(money_saved * 1000000, 2)  # At 1M requests/month
        }

# Global instance
context_optimizer = ContextOptimizer()
EOF

# API routes
cat > backend/api/routes/context_routes.py << 'EOF'
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from utils.token_counter import token_counter
from services.summarizer import summarizer
from services.context_optimizer import context_optimizer
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class TokenCountRequest(BaseModel):
    text: str = Field(..., description="Text to count tokens for")
    max_tokens: Optional[int] = Field(30000, description="Maximum token limit")

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    strategy: str = Field("extractive", description="Summarization strategy: extractive, abstractive, hybrid")
    target_ratio: float = Field(0.3, description="Target compression ratio (0.1-0.9)")

class OptimizeContextRequest(BaseModel):
    text: str = Field(..., description="Context to optimize")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens (uses default if not provided)")
    preserve_quality: bool = Field(True, description="Attempt to preserve quality during compression")

# Routes
@router.post("/count-tokens")
async def count_tokens(request: TokenCountRequest):
    """Count tokens in provided text"""
    try:
        analysis = token_counter.analyze(request.text, request.max_tokens)
        return {
            "success": True,
            "data": analysis
        }
    except Exception as e:
        logger.error(f"Token counting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """Summarize text using specified strategy"""
    try:
        if request.strategy not in ["extractive", "abstractive", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail="Strategy must be one of: extractive, abstractive, hybrid"
            )
        
        if not 0.1 <= request.target_ratio <= 0.9:
            raise HTTPException(
                status_code=400,
                detail="target_ratio must be between 0.1 and 0.9"
            )
        
        result = await summarizer.summarize(
            request.text,
            strategy=request.strategy,
            target_ratio=request.target_ratio
        )
        
        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-context")
async def optimize_context(request: OptimizeContextRequest):
    """Optimize context with intelligent compression"""
    try:
        result = await context_optimizer.optimize(
            request.text,
            max_tokens=request.max_tokens,
            preserve_quality=request.preserve_quality
        )
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Context optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "success": True,
        "data": {
            "service": "context-engineering",
            "version": "1.0.0",
            "strategies_available": ["extractive", "abstractive", "hybrid"],
            "default_max_tokens": 30000
        }
    }
EOF

cat > backend/api/__init__.py << 'EOF'
EOF

cat > backend/api/routes/__init__.py << 'EOF'
EOF

cat > backend/services/__init__.py << 'EOF'
EOF

cat > backend/utils/__init__.py << 'EOF'
EOF

# Simple test
cat > backend/tests/test_token_counter.py << 'EOF'
import pytest
from utils.token_counter import TokenCounter

def test_token_counter_basic():
    counter = TokenCounter()
    
    # Test basic counting
    text = "Hello, world!"
    count = counter.count(text)
    assert count > 0
    assert count < 10  # Should be small

def test_token_counter_analysis():
    counter = TokenCounter()
    
    text = "This is a test sentence." * 100
    analysis = counter.analyze(text, max_tokens=1000)
    
    assert "token_count" in analysis
    assert "usage_percent" in analysis
    assert "recommendation" in analysis

@pytest.mark.asyncio
async def test_empty_text():
    counter = TokenCounter()
    count = counter.count("")
    assert count == 0
EOF

echo "✓ Backend files created"

# ==================== FRONTEND FILES ====================

# Package.json
cat > frontend/package.json << 'EOF'
{
  "name": "vaia-l13-context-engineering",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.5",
    "recharts": "^2.10.3",
    "lucide-react": "^0.309.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# React App
cat > frontend/src/App.js << 'EOF'
import React, { useState } from 'react';
import TokenCounter from './components/TokenCounter';
import Summarizer from './components/Summarizer';
import ContextOptimizer from './components/ContextOptimizer';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('token-counter');
  const [stats, setStats] = useState({
    totalOptimizations: 0,
    totalTokensSaved: 0,
    averageCompressionRatio: 0,
    totalCostSavings: 0
  });

  const updateStats = (newData) => {
    setStats(prev => ({
      totalOptimizations: prev.totalOptimizations + 1,
      totalTokensSaved: prev.totalTokensSaved + (newData.tokens_saved || 0),
      averageCompressionRatio: newData.compression_ratio || prev.averageCompressionRatio,
      totalCostSavings: prev.totalCostSavings + (newData.money_saved || 0)
    }));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎯 VAIA L13: Context Engineering</h1>
        <p>Production-grade context management and optimization</p>
      </header>

      <Dashboard stats={stats} />

      <div className="tabs">
        <button 
          className={activeTab === 'token-counter' ? 'active' : ''}
          onClick={() => setActiveTab('token-counter')}
        >
          📊 Token Counter
        </button>
        <button 
          className={activeTab === 'summarizer' ? 'active' : ''}
          onClick={() => setActiveTab('summarizer')}
        >
          ✂️ Summarizer
        </button>
        <button 
          className={activeTab === 'optimizer' ? 'active' : ''}
          onClick={() => setActiveTab('optimizer')}
        >
          ⚡ Context Optimizer
        </button>
      </div>

      <div className="content">
        {activeTab === 'token-counter' && <TokenCounter />}
        {activeTab === 'summarizer' && <Summarizer onSummarize={updateStats} />}
        {activeTab === 'optimizer' && <ContextOptimizer onOptimize={updateStats} />}
      </div>

      <footer className="App-footer">
        <p>Building on L12: Few-Shot Prompting | Enabling L14: State Management</p>
        <p>Enterprise Context Management • Token Optimization • Cost Reduction</p>
      </footer>
    </div>
  );
}

export default App;
EOF

# Token Counter Component
cat > frontend/src/components/TokenCounter.js << 'EOF'
import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';

const API_URL = 'http://localhost:8000/api/v1';

function TokenCounter() {
  const [text, setText] = useState('');
  const [maxTokens, setMaxTokens] = useState(30000);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCount = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/count-tokens`, {
        text,
        max_tokens: maxTokens
      });

      setResult(response.data.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to count tokens');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = () => {
    if (!result) return null;
    
    if (result.recommendation === 'optimal') {
      return <CheckCircle className="status-icon success" />;
    } else if (result.recommendation === 'moderate') {
      return <AlertCircle className="status-icon warning" />;
    } else {
      return <AlertCircle className="status-icon error" />;
    }
  };

  return (
    <div className="component-container">
      <h2>Token Counter</h2>
      <p className="description">
        Analyze token usage and get optimization recommendations
      </p>

      <div className="input-group">
        <label>Text to Analyze:</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter or paste text here..."
          rows={10}
        />
      </div>

      <div className="input-group">
        <label>Max Tokens:</label>
        <input
          type="number"
          value={maxTokens}
          onChange={(e) => setMaxTokens(parseInt(e.target.value))}
          min={1000}
          max={100000}
        />
      </div>

      <button 
        onClick={handleCount}
        disabled={loading}
        className="primary-button"
      >
        {loading ? 'Analyzing...' : 'Count Tokens'}
      </button>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="result-header">
            {getStatusIcon()}
            <h3>Token Analysis</h3>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Token Count</span>
              <span className="metric-value">{result.token_count.toLocaleString()}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Usage</span>
              <span className="metric-value">{result.usage_percent}%</span>
            </div>

            <div className="metric">
              <span className="metric-label">Remaining</span>
              <span className="metric-value">{result.remaining_tokens.toLocaleString()}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Status</span>
              <span className={`metric-value status-${result.recommendation}`}>
                {result.recommendation.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="recommendation-box">
            <h4>Recommendation</h4>
            <p>
              {result.action === 'none' && 'Context size is optimal. No compression needed.'}
              {result.action === 'light_compression' && 'Consider light compression to reduce costs.'}
              {result.action === 'aggressive_compression' && 'Aggressive compression recommended to avoid limits.'}
            </p>
          </div>

          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ 
                width: `${Math.min(result.usage_percent, 100)}%`,
                backgroundColor: result.recommendation === 'optimal' ? '#4ade80' :
                                result.recommendation === 'moderate' ? '#fb923c' : '#ef4444'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default TokenCounter;
EOF

# Summarizer Component
cat > frontend/src/components/Summarizer.js << 'EOF'
import React, { useState } from 'react';
import axios from 'axios';
import { Scissors, Zap, GitMerge } from 'lucide-react';

const API_URL = 'http://localhost:8000/api/v1';

function Summarizer({ onSummarize }) {
  const [text, setText] = useState('');
  const [strategy, setStrategy] = useState('extractive');
  const [targetRatio, setTargetRatio] = useState(0.3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSummarize = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/summarize`, {
        text,
        strategy,
        target_ratio: targetRatio
      });

      const data = response.data.data;
      setResult(data);
      
      if (onSummarize) {
        onSummarize({
          tokens_saved: data.original_length - data.compressed_length,
          compression_ratio: data.compression_ratio,
          money_saved: ((data.original_length - data.compressed_length) / 1000) * 0.01
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to summarize');
    } finally {
      setLoading(false);
    }
  };

  const strategyIcons = {
    extractive: <Scissors className="strategy-icon" />,
    abstractive: <Zap className="strategy-icon" />,
    hybrid: <GitMerge className="strategy-icon" />
  };

  return (
    <div className="component-container">
      <h2>Text Summarizer</h2>
      <p className="description">
        Compress text using multiple strategies: extractive, abstractive, or hybrid
      </p>

      <div className="input-group">
        <label>Text to Summarize:</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter or paste text here..."
          rows={10}
        />
      </div>

      <div className="controls-row">
        <div className="input-group">
          <label>Strategy:</label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="extractive">Extractive (Fast)</option>
            <option value="abstractive">Abstractive (Quality)</option>
            <option value="hybrid">Hybrid (Balanced)</option>
          </select>
        </div>

        <div className="input-group">
          <label>Target Ratio: {(targetRatio * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.1"
            value={targetRatio}
            onChange={(e) => setTargetRatio(parseFloat(e.target.value))}
          />
        </div>
      </div>

      <button 
        onClick={handleSummarize}
        disabled={loading}
        className="primary-button"
      >
        {strategyIcons[strategy]}
        {loading ? 'Summarizing...' : `Summarize (${strategy})`}
      </button>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="result-header">
            <h3>Summary Result</h3>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Original</span>
              <span className="metric-value">{result.original_length.toLocaleString()} chars</span>
            </div>

            <div className="metric">
              <span className="metric-label">Compressed</span>
              <span className="metric-value">{result.compressed_length.toLocaleString()} chars</span>
            </div>

            <div className="metric">
              <span className="metric-label">Ratio</span>
              <span className="metric-value">{result.compression_ratio}x</span>
            </div>

            <div className="metric">
              <span className="metric-label">Time</span>
              <span className="metric-value">{result.compression_time_ms}ms</span>
            </div>
          </div>

          <div className="summary-box">
            <h4>Summary:</h4>
            <p>{result.summary}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default Summarizer;
EOF

# Context Optimizer Component  
cat > frontend/src/components/ContextOptimizer.js << 'EOF'
import React, { useState } from 'react';
import axios from 'axios';
import { Sparkles, DollarSign, TrendingDown } from 'lucide-react';

const API_URL = 'http://localhost:8000/api/v1';

function ContextOptimizer({ onOptimize }) {
  const [text, setText] = useState('');
  const [maxTokens, setMaxTokens] = useState(30000);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/optimize-context`, {
        text,
        max_tokens: maxTokens,
        preserve_quality: true
      });

      const data = response.data.data;
      setResult(data);
      
      if (onOptimize && data.cost_savings) {
        onOptimize({
          tokens_saved: data.cost_savings.tokens_saved,
          compression_ratio: data.compression_ratio,
          money_saved: data.cost_savings.money_saved_per_request
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to optimize context');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="component-container">
      <h2>Context Optimizer</h2>
      <p className="description">
        Intelligent context optimization with automatic strategy selection
      </p>

      <div className="input-group">
        <label>Context to Optimize:</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter or paste context here..."
          rows={10}
        />
      </div>

      <div className="input-group">
        <label>Max Tokens:</label>
        <input
          type="number"
          value={maxTokens}
          onChange={(e) => setMaxTokens(parseInt(e.target.value))}
          min={1000}
          max={100000}
        />
      </div>

      <button 
        onClick={handleOptimize}
        disabled={loading}
        className="primary-button"
      >
        <Sparkles className="button-icon" />
        {loading ? 'Optimizing...' : 'Optimize Context'}
      </button>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="result-header">
            <h3>Optimization Result</h3>
            <span className={`status-badge ${result.quality_preserved ? 'success' : 'warning'}`}>
              {result.quality_preserved ? 'Quality Preserved' : 'Best Effort'}
            </span>
          </div>

          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Original Tokens</span>
              <span className="metric-value">{result.original_tokens.toLocaleString()}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Optimized Tokens</span>
              <span className="metric-value">{result.optimized_tokens.toLocaleString()}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Compression</span>
              <span className="metric-value success">{result.compression_ratio.toFixed(2)}x</span>
            </div>

            <div className="metric">
              <span className="metric-label">Strategy</span>
              <span className="metric-value">{result.strategy}</span>
            </div>
          </div>

          {result.cost_savings && (
            <div className="savings-card">
              <h4><DollarSign className="icon" /> Cost Savings</h4>
              <div className="savings-grid">
                <div className="saving-item">
                  <span className="label">Tokens Saved</span>
                  <span className="value">{result.cost_savings.tokens_saved.toLocaleString()}</span>
                </div>
                <div className="saving-item">
                  <span className="label">% Reduction</span>
                  <span className="value success">
                    <TrendingDown className="icon-small" />
                    {result.cost_savings.percent_reduction}%
                  </span>
                </div>
                <div className="saving-item">
                  <span className="label">Per Request</span>
                  <span className="value">${result.cost_savings.money_saved_per_request.toFixed(6)}</span>
                </div>
                <div className="saving-item">
                  <span className="label">Monthly (1M req)</span>
                  <span className="value success">${result.cost_savings.projected_monthly_savings.toLocaleString()}</span>
                </div>
              </div>
            </div>
          )}

          <div className="optimized-text-box">
            <h4>Optimized Context:</h4>
            <p>{result.optimized_text}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default ContextOptimizer;
EOF

# Dashboard Component
cat > frontend/src/components/Dashboard.js << 'EOF'
import React from 'react';
import { Activity, TrendingDown, DollarSign, Zap } from 'lucide-react';

function Dashboard({ stats }) {
  return (
    <div className="dashboard">
      <div className="stat-card">
        <Activity className="stat-icon" />
        <div className="stat-content">
          <span className="stat-label">Total Optimizations</span>
          <span className="stat-value">{stats.totalOptimizations}</span>
        </div>
      </div>

      <div className="stat-card">
        <TrendingDown className="stat-icon success" />
        <div className="stat-content">
          <span className="stat-label">Tokens Saved</span>
          <span className="stat-value">{stats.totalTokensSaved.toLocaleString()}</span>
        </div>
      </div>

      <div className="stat-card">
        <Zap className="stat-icon warning" />
        <div className="stat-content">
          <span className="stat-label">Avg Compression</span>
          <span className="stat-value">{stats.averageCompressionRatio.toFixed(2)}x</span>
        </div>
      </div>

      <div className="stat-card">
        <DollarSign className="stat-icon" />
        <div className="stat-content">
          <span className="stat-label">Cost Savings</span>
          <span className="stat-value">${stats.totalCostSavings.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
EOF

# App CSS
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
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.App-header {
  text-align: center;
  color: white;
  margin-bottom: 30px;
}

.App-header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.App-header p {
  font-size: 1.1rem;
  opacity: 0.9;
}

.dashboard {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.stat-card {
  background: white;
  border-radius: 12px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-icon {
  width: 40px;
  height: 40px;
  color: #667eea;
}

.stat-icon.success {
  color: #10b981;
}

.stat-icon.warning {
  color: #f59e0b;
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-label {
  font-size: 0.9rem;
  color: #6b7280;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 1.8rem;
  font-weight: bold;
  color: #1f2937;
}

.tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  background: white;
  padding: 10px;
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.tabs button {
  flex: 1;
  padding: 12px 24px;
  border: none;
  background: transparent;
  color: #6b7280;
  font-size: 1rem;
  font-weight: 500;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.tabs button:hover {
  background: #f3f4f6;
}

.tabs button.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
}

.content {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  min-height: 600px;
}

.component-container h2 {
  color: #1f2937;
  margin-bottom: 10px;
  font-size: 1.8rem;
}

.description {
  color: #6b7280;
  margin-bottom: 30px;
  font-size: 1.1rem;
}

.input-group {
  margin-bottom: 20px;
}

.input-group label {
  display: block;
  color: #374151;
  font-weight: 500;
  margin-bottom: 8px;
}

.input-group textarea,
.input-group input[type="number"],
.input-group input[type="text"],
.input-group select {
  width: 100%;
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  transition: border-color 0.2s;
}

.input-group textarea {
  resize: vertical;
  font-family: 'Courier New', monospace;
}

.input-group textarea:focus,
.input-group input:focus,
.input-group select:focus {
  outline: none;
  border-color: #667eea;
}

.input-group input[type="range"] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: #e5e7eb;
  outline: none;
}

.input-group input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #667eea;
  cursor: pointer;
}

.controls-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.primary-button {
  padding: 14px 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.primary-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.primary-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.button-icon {
  width: 20px;
  height: 20px;
}

.error-message {
  background: #fee2e2;
  color: #991b1b;
  padding: 12px;
  border-radius: 8px;
  margin-top: 20px;
}

.result-card {
  margin-top: 30px;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  padding: 24px;
  background: #f9fafb;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}

.result-header h3 {
  color: #1f2937;
  font-size: 1.5rem;
}

.status-icon {
  width: 28px;
  height: 28px;
}

.status-icon.success {
  color: #10b981;
}

.status-icon.warning {
  color: #f59e0b;
}

.status-icon.error {
  color: #ef4444;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  font-size: 0.9rem;
  color: #6b7280;
  font-weight: 500;
}

.metric-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #1f2937;
}

.metric-value.success {
  color: #10b981;
}

.metric-value.status-optimal {
  color: #10b981;
}

.metric-value.status-moderate {
  color: #f59e0b;
}

.metric-value.status-critical {
  color: #ef4444;
}

.recommendation-box,
.summary-box,
.optimized-text-box {
  background: white;
  border-radius: 8px;
  padding: 16px;
  margin-top: 20px;
}

.recommendation-box h4,
.summary-box h4,
.optimized-text-box h4 {
  color: #374151;
  margin-bottom: 12px;
}

.recommendation-box p,
.summary-box p,
.optimized-text-box p {
  color: #6b7280;
  line-height: 1.6;
}

.progress-bar {
  width: 100%;
  height: 12px;
  background: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 20px;
}

.progress-fill {
  height: 100%;
  transition: width 0.3s ease;
  border-radius: 6px;
}

.savings-card {
  background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
}

.savings-card h4 {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #065f46;
  margin-bottom: 16px;
}

.savings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.saving-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.saving-item .label {
  font-size: 0.85rem;
  color: #047857;
  font-weight: 500;
}

.saving-item .value {
  font-size: 1.3rem;
  font-weight: bold;
  color: #065f46;
  display: flex;
  align-items: center;
  gap: 4px;
}

.saving-item .value.success {
  color: #059669;
}

.icon {
  width: 20px;
  height: 20px;
}

.icon-small {
  width: 16px;
  height: 16px;
}

.status-badge {
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-left: auto;
}

.status-badge.success {
  background: #d1fae5;
  color: #065f46;
}

.status-badge.warning {
  background: #fed7aa;
  color: #92400e;
}

.strategy-icon {
  width: 20px;
  height: 20px;
}

.App-footer {
  text-align: center;
  color: white;
  margin-top: 40px;
  padding: 20px;
  opacity: 0.9;
}

.App-footer p {
  margin: 8px 0;
  font-size: 0.95rem;
}

@media (max-width: 768px) {
  .App {
    padding: 10px;
  }

  .App-header h1 {
    font-size: 1.8rem;
  }

  .controls-row {
    grid-template-columns: 1fr;
  }

  .tabs {
    flex-direction: column;
  }
  
  .metrics-grid,
  .savings-grid {
    grid-template-columns: 1fr 1fr;
  }
}
EOF

# Index files
cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

cat > frontend/src/index.css << 'EOF'
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="VAIA L13: Context Engineering" />
    <title>VAIA L13: Context Engineering</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

echo "✓ Frontend files created"

# ==================== DOCKER FILES ====================

cat > docker/Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application
COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker/Dockerfile.frontend << 'EOF'
FROM node:18-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "start"]
EOF

cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ../backend:/app
    restart: unless-stopped

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ../frontend/src:/app/src
    restart: unless-stopped
EOF

echo "✓ Docker files created"

# ==================== HELPER SCRIPTS ====================

cat > build.sh << 'EOF'
#!/bin/bash
echo "Building VAIA L13: Context Engineering..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Building with Docker..."
    cd docker
    docker-compose build
    echo "✓ Docker build complete"
else
    echo "Docker not found, setting up local environment..."
    
    # Backend setup
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    cd ..
    
    # Frontend setup
    cd frontend
    npm install
    cd ..
    
    echo "✓ Local build complete"
fi
EOF

cat > start.sh << 'EOF'
#!/bin/bash
echo "Starting VAIA L13: Context Engineering..."

if command -v docker &> /dev/null && [ -f "docker/docker-compose.yml" ]; then
    echo "Starting with Docker..."
    cd docker
    docker-compose up -d
    echo "✓ Services started"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
else
    echo "Starting local services..."
    
    # Start backend
    cd backend
    source venv/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo "✓ Services started"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
fi
EOF

cat > stop.sh << 'EOF'
#!/bin/bash
echo "Stopping VAIA L13: Context Engineering..."

if command -v docker &> /dev/null && [ -f "docker/docker-compose.yml" ]; then
    cd docker
    docker-compose down
    echo "✓ Docker services stopped"
else
    pkill -f "uvicorn main:app"
    pkill -f "react-scripts start"
    echo "✓ Local services stopped"
fi
EOF

cat > test.sh << 'EOF'
#!/bin/bash
echo "Testing VAIA L13: Context Engineering..."

cd backend
source venv/bin/activate 2>/dev/null || true
pytest tests/ -v
cd ..

echo "✓ Tests complete"
EOF

chmod +x build.sh start.sh stop.sh test.sh

echo "✓ Helper scripts created"

# ==================== README ====================

cat > README.md << 'EOF'
# VAIA L13: Context Engineering

Production-grade context management and optimization for enterprise AI agents.

## Overview

This lesson implements intelligent context window management using:
- Token counting with tiktoken
- Multi-strategy summarization (extractive, abstractive, hybrid)
- Automatic context optimization
- Real-time cost analysis

## Quick Start

### Using Docker (Recommended)
```bash
./build.sh
./start.sh
```

### Local Development
```bash
# Install dependencies
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Architecture

- **Backend**: FastAPI + Gemini AI + tiktoken
- **Frontend**: React with real-time monitoring
- **Context Manager**: Intelligent optimization pipeline

## Features

1. **Token Counter**: Precise token analysis with recommendations
2. **Summarizer**: Multiple compression strategies
3. **Context Optimizer**: Automatic strategy selection and optimization
4. **Dashboard**: Real-time metrics and cost savings

## API Endpoints

- `POST /api/v1/count-tokens` - Analyze token usage
- `POST /api/v1/summarize` - Compress text
- `POST /api/v1/optimize-context` - Intelligent optimization
- `GET /health` - Service health check

## Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Building on L12

Extends few-shot prompting with context management for token-efficient examples.

## Enabling L14

Provides context optimization layer for state serialization and persistence.

## License

Part of VAIA 90-Lesson Curriculum
EOF

echo "✓ README created"

# Create lesson metadata
cat > lesson_metadata.json << 'EOF'
{
  "lesson_number": 13,
  "module": 2,
  "module_name": "Core Agent Foundations",
  "lesson_title": "Introduction to Context Engineering",
  "previous_lesson": {
    "number": 12,
    "title": "Prompt Engineering Mastery: Few-Shot",
    "components_available": [
      "few_shot_classifier",
      "prompt_templates",
      "example_management"
    ]
  },
  "next_lesson": {
    "number": 14,
    "title": "State Management for Agents",
    "prerequisites_provided": [
      "token_counter",
      "context_optimizer",
      "compression_strategies"
    ]
  },
  "difficulty": "Intermediate",
  "estimated_time": "3 hours",
  "skills_gained": [
    "Token counting and analysis",
    "Multi-strategy text summarization",
    "Context window optimization",
    "Cost-aware LLM usage",
    "Production context management"
  ],
  "vaia_components_built": [
    "TokenCounter",
    "SummarizerService",
    "ContextOptimizer",
    "React Dashboard"
  ]
}
EOF

echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "Project: $PROJECT_NAME"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. ./build.sh"
echo "3. ./start.sh"
echo "4. Open http://localhost:3000"
echo ""
echo "Building on L12: Few-Shot Prompting"
echo "Enabling L14: State Management"
echo ""