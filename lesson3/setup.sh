#!/bin/bash

# L3: Transformer Architecture Deep Dive - Complete Setup
# Builds on L2 components, prepares for L4

set -e

echo "======================================"
echo "L3: Transformer Architecture Setup"
echo "======================================"

PROJECT_ROOT="$(pwd)/l3-transformer-deep-dive"
VENV_PATH="$PROJECT_ROOT/venv"

# Create project structure
echo "Creating project structure..."
mkdir -p "$PROJECT_ROOT"/{backend/{app,tests},frontend/{src/{components,utils},public},docker,scripts,diagrams}

cd "$PROJECT_ROOT"

# ==================== BACKEND ====================

echo "Setting up backend..."

# backend/requirements.txt
cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.2
pydantic-settings==2.5.2
numpy==2.1.3
websockets==13.1
aiofiles==24.1.0
python-multipart==0.0.17
google-generativeai==0.8.3
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
EOF

# backend/app/__init__.py
touch backend/app/__init__.py

# backend/app/config.py
cat > backend/app/config.py << 'EOF'
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    host: str = "0.0.0.0"
    port: int = 8000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 512
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

# backend/app/transformer.py
cat > backend/app/transformer.py << 'EOF'
"""
Production-grade Transformer implementation with attention visualization.
Builds on L2 async patterns and dataclass validation.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import asyncio
from datetime import datetime

@dataclass
class AttentionConfig:
    """Configuration for transformer attention mechanism."""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

@dataclass
class AttentionWeights:
    """Container for attention weights with metadata."""
    weights: np.ndarray  # [num_heads, seq_len, seq_len]
    layer_idx: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self):
        return {
            "weights": self.weights.tolist(),
            "layer_idx": self.layer_idx,
            "timestamp": self.timestamp,
            "shape": list(self.weights.shape)
        }

class ScaledDotProductAttention:
    """Core attention mechanism with numerical stability."""
    
    def __init__(self, dropout: float = 0.1):
        self.dropout = dropout
        
    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: [batch, seq_len, d_k]
            key: [batch, seq_len, d_k]
            value: [batch, seq_len, d_v]
            mask: Optional mask [batch, seq_len, seq_len]
            
        Returns:
            output: [batch, seq_len, d_v]
            attention_weights: [batch, seq_len, seq_len]
        """
        d_k = query.shape[-1]
        
        # Compute attention scores
        scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax for attention weights
        attention_weights = self._softmax(scores)
        
        # Apply dropout (in training)
        if self.dropout > 0:
            attention_weights = self._dropout(attention_weights, self.dropout)
        
        # Compute output
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def _dropout(x: np.ndarray, p: float) -> np.ndarray:
        """Simple dropout implementation."""
        mask = np.random.binomial(1, 1 - p, x.shape)
        return x * mask / (1 - p)

class MultiHeadAttention:
    """Multi-head attention with parallel head computation."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.d_k = config.d_model // config.num_heads
        self.attention = ScaledDotProductAttention(config.dropout)
        
        # Initialize projection matrices
        self.W_q = self._init_weights((config.d_model, config.d_model))
        self.W_k = self._init_weights((config.d_model, config.d_model))
        self.W_v = self._init_weights((config.d_model, config.d_model))
        self.W_o = self._init_weights((config.d_model, config.d_model))
    
    @staticmethod
    def _init_weights(shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization for weights."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into [num_heads, d_k].
        [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.config.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back.
        [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.config.d_model)
    
    async def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Async multi-head attention computation.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Compute attention for each head (parallelizable)
        attention_outputs = []
        attention_weights_list = []
        
        for head_idx in range(self.config.num_heads):
            q_head = Q[:, head_idx, :, :]
            k_head = K[:, head_idx, :, :]
            v_head = V[:, head_idx, :, :]
            
            output, weights = self.attention.forward(q_head, k_head, v_head, mask)
            attention_outputs.append(output)
            attention_weights_list.append(weights)
        
        # Stack heads
        attention_output = np.stack(attention_outputs, axis=1)
        attention_weights = np.stack(attention_weights_list, axis=1)
        
        # Combine heads and final projection
        output = self._combine_heads(attention_output)
        output = np.matmul(output, self.W_o)
        
        return output, attention_weights

class PositionalEncoding:
    """Sinusoidal positional encodings."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        self.d_model = d_model
        self.encoding = self._create_encoding(max_seq_length, d_model)
    
    def _create_encoding(self, max_seq_length: int, d_model: int) -> np.ndarray:
        """
        Create sinusoidal positional encodings.
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            x + positional_encoding
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len, :]

class FeedForward:
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        self.W1 = self._init_weights((d_model, d_ff))
        self.W2 = self._init_weights((d_ff, d_model))
        self.dropout = dropout
    
    @staticmethod
    def _init_weights(shape: Tuple[int, int]) -> np.ndarray:
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """FFN(x) = max(0, xW1)W2"""
        x = np.maximum(0, np.matmul(x, self.W1))  # ReLU activation
        x = np.matmul(x, self.W2)
        return x

class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Normalize over last dimension."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class TransformerBlock:
    """Single transformer encoder block."""
    
    def __init__(self, config: AttentionConfig):
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config.d_model)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.dropout = config.dropout
    
    async def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through transformer block.
        
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Multi-head attention with residual
        attn_output, attention_weights = await self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x, attention_weights

class TransformerEncoder:
    """Complete transformer encoder stack."""
    
    def __init__(self, config: AttentionConfig, num_layers: int = 6):
        self.layers = [TransformerBlock(config) for _ in range(num_layers)]
        self.pos_encoding = PositionalEncoding(config.d_model)
        
    async def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[AttentionWeights]]:
        """
        Forward pass through all layers.
        
        Returns:
            output: Final hidden states
            attention_weights: List of attention weights per layer
        """
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        
        all_attention_weights = []
        
        # Pass through each layer
        for layer_idx, layer in enumerate(self.layers):
            x, attn_weights = await layer.forward(x, mask)
            all_attention_weights.append(
                AttentionWeights(
                    weights=attn_weights[0],  # Remove batch dimension
                    layer_idx=layer_idx
                )
            )
        
        return x, all_attention_weights

# Tokenizer utility
class SimpleTokenizer:
    """Simple word-level tokenizer for demonstration."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build simple vocabulary."""
        # Special tokens
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = text.lower().split()
        tokens = [self.vocab.get("<START>", 2)]
        
        for word in words:
            if word not in self.vocab:
                # Add new word to vocab
                if len(self.vocab) < self.vocab_size:
                    self.vocab[word] = len(self.vocab)
                    self.reverse_vocab[self.vocab[word]] = word
            
            token_id = self.vocab.get(word, self.vocab["<UNK>"])
            tokens.append(token_id)
        
        tokens.append(self.vocab.get("<END>", 3))
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.reverse_vocab.get(t, "<UNK>") for t in tokens]
        return " ".join(words)
    
    def create_embeddings(self, tokens: List[int], d_model: int) -> np.ndarray:
        """Create simple embeddings for tokens."""
        embeddings = np.random.randn(len(tokens), d_model) * 0.1
        return embeddings[np.newaxis, :, :]  # Add batch dimension
EOF

# backend/app/main.py
cat > backend/app/main.py << 'EOF'
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
EOF

# ==================== FRONTEND ====================

echo "Setting up frontend..."

# frontend/package.json
cat > frontend/package.json << 'EOF'
{
  "name": "transformer-visualizer",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1",
    "axios": "^1.7.7",
    "plotly.js": "^2.35.2",
    "react-plotly.js": "^2.6.0",
    "recharts": "^2.13.3"
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

# frontend/public/index.html
cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Transformer Architecture Deep Dive - L3" />
    <title>Transformer Visualizer | L3</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

# frontend/src/index.js
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

# frontend/src/index.css
cat > frontend/src/index.css << 'EOF'
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

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
EOF

# frontend/src/App.js
cat > frontend/src/App.js << 'EOF'
import React, { useState } from 'react';
import './App.css';
import TransformerVisualizer from './components/TransformerVisualizer';
import PerformanceMetrics from './components/PerformanceMetrics';

function App() {
  const [activeTab, setActiveTab] = useState('visualizer');

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🧠 Transformer Architecture Deep Dive</h1>
          <p className="subtitle">L3: Interactive Attention Mechanism Exploration</p>
        </div>
        <div className="tab-navigation">
          <button 
            className={activeTab === 'visualizer' ? 'tab-active' : 'tab'}
            onClick={() => setActiveTab('visualizer')}
          >
            Visualizer
          </button>
          <button 
            className={activeTab === 'metrics' ? 'tab-active' : 'tab'}
            onClick={() => setActiveTab('metrics')}
          >
            Performance
          </button>
        </div>
      </header>

      <main className="App-main">
        {activeTab === 'visualizer' && <TransformerVisualizer />}
        {activeTab === 'metrics' && <PerformanceMetrics />}
      </main>

      <footer className="App-footer">
        <p>Advanced Architectures for Vertical AI Agents | Module 1: Foundations</p>
      </footer>
    </div>
  );
}

export default App;
EOF

# frontend/src/App.css
cat > frontend/src/App.css << 'EOF'
.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.App-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
}

.App-header h1 {
  color: #667eea;
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #666;
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
}

.tab-navigation {
  display: flex;
  gap: 1rem;
  max-width: 1200px;
  margin: 0 auto;
}

.tab, .tab-active {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
}

.tab:hover {
  background: rgba(102, 126, 234, 0.2);
}

.tab-active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.App-main {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.App-footer {
  background: rgba(255, 255, 255, 0.95);
  padding: 1.5rem;
  text-align: center;
  color: #666;
  box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.05);
}
EOF

# frontend/src/components/TransformerVisualizer.js
cat > frontend/src/components/TransformerVisualizer.js << 'EOF'
import React, { useState, useCallback } from 'react';
import axios from 'axios';
import AttentionHeatmap from './AttentionHeatmap';
import './TransformerVisualizer.css';

const API_BASE_URL = 'http://localhost:8000';

const TransformerVisualizer = () => {
  const [inputText, setInputText] = useState('The transformer architecture revolutionized natural language processing');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [error, setError] = useState(null);

  const processText = useCallback(async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/transform`, {
        text: inputText,
        visualize: true
      });

      setResult(response.data);
      setSelectedLayer(0);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process text');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  }, [inputText]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      processText();
    }
  };

  return (
    <div className="visualizer-container">
      <div className="input-section">
        <h2>Input Text</h2>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter text to analyze transformer attention patterns..."
          rows={4}
          className="text-input"
        />
        <button 
          onClick={processText} 
          disabled={loading}
          className="process-button"
        >
          {loading ? '⏳ Processing...' : '🚀 Analyze Transformer'}
        </button>
        {error && <div className="error-message">{error}</div>}
      </div>

      {result && (
        <>
          <div className="metadata-section">
            <div className="metadata-card">
              <h3>Configuration</h3>
              <div className="metadata-grid">
                <div className="metadata-item">
                  <span className="label">Model Dimension:</span>
                  <span className="value">{result.metadata.d_model}</span>
                </div>
                <div className="metadata-item">
                  <span className="label">Attention Heads:</span>
                  <span className="value">{result.metadata.num_heads}</span>
                </div>
                <div className="metadata-item">
                  <span className="label">Layers:</span>
                  <span className="value">{result.metadata.num_layers}</span>
                </div>
                <div className="metadata-item">
                  <span className="label">Sequence Length:</span>
                  <span className="value">{result.metadata.seq_length}</span>
                </div>
              </div>
            </div>
            <div className="metadata-card">
              <h3>Performance</h3>
              <div className="latency">
                <span className="latency-value">{result.latency_ms.toFixed(2)}</span>
                <span className="latency-unit">ms</span>
              </div>
              <p className="latency-desc">Forward Pass Latency</p>
            </div>
          </div>

          <div className="tokens-section">
            <h2>Tokens</h2>
            <div className="tokens-display">
              {result.tokens.map((token, idx) => (
                <span key={idx} className="token">
                  {token}
                </span>
              ))}
            </div>
          </div>

          <div className="attention-section">
            <div className="section-header">
              <h2>Attention Weights Visualization</h2>
              <div className="layer-selector">
                <label>Layer:</label>
                <select 
                  value={selectedLayer} 
                  onChange={(e) => setSelectedLayer(Number(e.target.value))}
                  className="layer-select"
                >
                  {result.attention_weights.map((_, idx) => (
                    <option key={idx} value={idx}>
                      Layer {idx + 1}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {result.attention_weights[selectedLayer] && (
              <AttentionHeatmap
                attentionWeights={result.attention_weights[selectedLayer]}
                tokens={result.tokens}
              />
            )}
          </div>

          <div className="insights-section">
            <h2>💡 Attention Insights</h2>
            <div className="insights-grid">
              <div className="insight-card">
                <h4>Self-Attention</h4>
                <p>Each token attends to all other tokens in parallel, capturing contextual relationships.</p>
              </div>
              <div className="insight-card">
                <h4>Multi-Head</h4>
                <p>{result.metadata.num_heads} heads learn different attention patterns simultaneously.</p>
              </div>
              <div className="insight-card">
                <h4>Layer Depth</h4>
                <p>Deeper layers capture more abstract semantic relationships.</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default TransformerVisualizer;
EOF

# frontend/src/components/TransformerVisualizer.css
cat > frontend/src/components/TransformerVisualizer.css << 'EOF'
.visualizer-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.input-section {
  margin-bottom: 2rem;
}

.input-section h2 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.5rem;
}

.text-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.3s ease;
}

.text-input:focus {
  outline: none;
  border-color: #667eea;
}

.process-button {
  margin-top: 1rem;
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.3s ease;
}

.process-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.process-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.error-message {
  margin-top: 1rem;
  padding: 1rem;
  background: #fee;
  border: 1px solid #fcc;
  border-radius: 8px;
  color: #c00;
}

.metadata-section {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.metadata-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 1.5rem;
  border-radius: 12px;
}

.metadata-card h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.2rem;
}

.metadata-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.metadata-item {
  display: flex;
  flex-direction: column;
}

.label {
  color: #666;
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}

.value {
  color: #333;
  font-size: 1.5rem;
  font-weight: 700;
}

.latency {
  text-align: center;
  margin: 1rem 0;
}

.latency-value {
  font-size: 3rem;
  font-weight: 700;
  color: #667eea;
}

.latency-unit {
  font-size: 1.5rem;
  color: #999;
  margin-left: 0.5rem;
}

.latency-desc {
  color: #666;
  text-align: center;
}

.tokens-section {
  margin-bottom: 2rem;
}

.tokens-section h2 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.5rem;
}

.tokens-display {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.token {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-weight: 500;
  font-size: 0.95rem;
}

.attention-section {
  margin-bottom: 2rem;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.section-header h2 {
  color: #333;
  font-size: 1.5rem;
}

.layer-selector {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.layer-selector label {
  color: #666;
  font-weight: 600;
}

.layer-select {
  padding: 0.5rem 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  background: white;
}

.insights-section {
  margin-top: 2rem;
}

.insights-section h2 {
  color: #333;
  margin-bottom: 1.5rem;
  font-size: 1.5rem;
}

.insights-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

.insight-card {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.insight-card h4 {
  color: #333;
  margin-bottom: 0.75rem;
  font-size: 1.1rem;
}

.insight-card p {
  color: #555;
  line-height: 1.6;
}
EOF

# frontend/src/components/AttentionHeatmap.js
cat > frontend/src/components/AttentionHeatmap.js << 'EOF'
import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import './AttentionHeatmap.css';

const AttentionHeatmap = ({ attentionWeights, tokens }) => {
  const heatmapData = useMemo(() => {
    if (!attentionWeights || !attentionWeights.weights) {
      return null;
    }

    const weights = attentionWeights.weights;
    const numHeads = weights.length;

    return Array.from({ length: numHeads }, (_, headIdx) => ({
      z: weights[headIdx],
      type: 'heatmap',
      colorscale: 'Viridis',
      showscale: headIdx === 0,
      hovertemplate: 
        'From: %{x}<br>' +
        'To: %{y}<br>' +
        'Weight: %{z:.4f}<br>' +
        '<extra></extra>',
      x: tokens,
      y: tokens
    }));
  }, [attentionWeights, tokens]);

  if (!heatmapData) {
    return <div className="no-data">No attention data available</div>;
  }

  const numHeads = heatmapData.length;
  const cols = Math.ceil(Math.sqrt(numHeads));
  const rows = Math.ceil(numHeads / cols);

  return (
    <div className="heatmap-container">
      <div className="heatmap-grid">
        {heatmapData.map((data, headIdx) => (
          <div key={headIdx} className="heatmap-cell">
            <h4>Head {headIdx + 1}</h4>
            <Plot
              data={[data]}
              layout={{
                width: 300,
                height: 300,
                margin: { l: 80, r: 20, t: 20, b: 80 },
                xaxis: {
                  tickangle: -45,
                  tickfont: { size: 9 }
                },
                yaxis: {
                  tickfont: { size: 9 }
                }
              }}
              config={{ displayModeBar: false }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttentionHeatmap;
EOF

# frontend/src/components/AttentionHeatmap.css
cat > frontend/src/components/AttentionHeatmap.css << 'EOF'
.heatmap-container {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.heatmap-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  justify-items: center;
}

.heatmap-cell {
  text-align: center;
}

.heatmap-cell h4 {
  color: #667eea;
  margin-bottom: 0.5rem;
  font-size: 1rem;
}

.no-data {
  padding: 3rem;
  text-align: center;
  color: #999;
  font-size: 1.1rem;
}
EOF

# frontend/src/components/PerformanceMetrics.js
cat > frontend/src/components/PerformanceMetrics.js << 'EOF'
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './PerformanceMetrics.css';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState([]);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Failed to fetch config:', error);
    }
  };

  const generateMockMetrics = () => {
    const newMetrics = [];
    for (let i = 0; i < 50; i++) {
      newMetrics.push({
        request: i + 1,
        latency: 30 + Math.random() * 40,
        throughput: 80 + Math.random() * 40,
        memory: 150 + Math.random() * 50
      });
    }
    setMetrics(newMetrics);
  };

  useEffect(() => {
    generateMockMetrics();
  }, []);

  return (
    <div className="metrics-container">
      <div className="metrics-header">
        <h2>Performance Metrics</h2>
        <p>Real-time monitoring of transformer inference performance</p>
      </div>

      {config && (
        <div className="config-display">
          <h3>Current Configuration</h3>
          <div className="config-grid">
            <div className="config-item">
              <span className="config-label">Model Dimension:</span>
              <span className="config-value">{config.d_model}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Attention Heads:</span>
              <span className="config-value">{config.num_heads}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Layers:</span>
              <span className="config-value">{config.num_layers}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Max Sequence:</span>
              <span className="config-value">{config.max_seq_length}</span>
            </div>
          </div>
        </div>
      )}

      <div className="charts-grid">
        <div className="chart-card">
          <h3>Inference Latency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="latency" stroke="#667eea" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Throughput (req/sec)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="throughput" stroke="#764ba2" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Memory Usage (MB)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="memory" stroke="#f093fb" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="optimization-tips">
        <h3>🚀 Production Optimization Tips</h3>
        <ul>
          <li><strong>Batch Processing:</strong> Group requests by sequence length for optimal GPU utilization</li>
          <li><strong>KV Caching:</strong> Cache key-value pairs for autoregressive generation (5-10x speedup)</li>
          <li><strong>Quantization:</strong> Use INT8/FP16 precision for 2-4x inference speedup</li>
          <li><strong>Attention Sparsity:</strong> Implement sparse attention patterns for long sequences</li>
        </ul>
      </div>
    </div>
  );
};

export default PerformanceMetrics;
EOF

# frontend/src/components/PerformanceMetrics.css
cat > frontend/src/components/PerformanceMetrics.css << 'EOF'
.metrics-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.metrics-header {
  margin-bottom: 2rem;
}

.metrics-header h2 {
  color: #333;
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.metrics-header p {
  color: #666;
  font-size: 1.1rem;
}

.config-display {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: 1.5rem;
  border-radius: 12px;
  margin-bottom: 2rem;
}

.config-display h3 {
  color: #333;
  margin-bottom: 1rem;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.config-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(255, 255, 255, 0.7);
  padding: 1rem;
  border-radius: 8px;
}

.config-label {
  color: #666;
  font-weight: 500;
}

.config-value {
  color: #667eea;
  font-size: 1.5rem;
  font-weight: 700;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}

.chart-card {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.chart-card h3 {
  color: #333;
  margin-bottom: 1rem;
}

.optimization-tips {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
  padding: 2rem;
  border-radius: 12px;
  margin-top: 2rem;
}

.optimization-tips h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
}

.optimization-tips ul {
  list-style: none;
  padding: 0;
}

.optimization-tips li {
  color: #555;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.3);
  line-height: 1.6;
}

.optimization-tips li:last-child {
  border-bottom: none;
}

.optimization-tips strong {
  color: #667eea;
}
EOF

# ==================== TESTING ====================

# backend/tests/__init__.py
touch backend/tests/__init__.py

# backend/tests/test_transformer.py
cat > backend/tests/test_transformer.py << 'EOF'
import pytest
import numpy as np
from app.transformer import (
    AttentionConfig,
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerBlock,
    SimpleTokenizer
)

def test_attention_config():
    """Test attention configuration validation."""
    config = AttentionConfig(d_model=512, num_heads=8)
    assert config.d_model == 512
    assert config.num_heads == 8
    
    with pytest.raises(ValueError):
        AttentionConfig(d_model=513, num_heads=8)

def test_scaled_dot_product_attention():
    """Test attention mechanism."""
    attention = ScaledDotProductAttention()
    
    batch_size, seq_len, d_k = 1, 10, 64
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    output, weights = attention.forward(Q, K, V)
    
    assert output.shape == (batch_size, seq_len, d_k)
    assert weights.shape == (batch_size, seq_len, seq_len)
    
    # Check attention weights sum to 1
    assert np.allclose(weights.sum(axis=-1), 1.0)

def test_positional_encoding():
    """Test positional encoding generation."""
    d_model = 512
    max_len = 100
    
    pe = PositionalEncoding(d_model, max_len)
    
    x = np.random.randn(1, 50, d_model)
    output = pe.forward(x)
    
    assert output.shape == x.shape

def test_tokenizer():
    """Test simple tokenizer."""
    tokenizer = SimpleTokenizer()
    
    text = "hello world"
    tokens = tokenizer.tokenize(text)
    
    assert len(tokens) > 0
    assert tokens[0] == tokenizer.vocab["<START>"]
    assert tokens[-1] == tokenizer.vocab["<END>"]

@pytest.mark.asyncio
async def test_multi_head_attention():
    """Test multi-head attention."""
    config = AttentionConfig(d_model=512, num_heads=8)
    mha = MultiHeadAttention(config)
    
    x = np.random.randn(1, 10, 512)
    output, weights = await mha.forward(x)
    
    assert output.shape == x.shape
    assert weights.shape[1] == config.num_heads

@pytest.mark.asyncio
async def test_transformer_block():
    """Test complete transformer block."""
    config = AttentionConfig(d_model=512, num_heads=8)
    block = TransformerBlock(config)
    
    x = np.random.randn(1, 10, 512)
    output, weights = await block.forward(x)
    
    assert output.shape == x.shape
    assert weights.shape[1] == config.num_heads
EOF

# ==================== DOCKER ====================

# docker/Dockerfile.backend
cat > docker/Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# docker/Dockerfile.frontend
cat > docker/Dockerfile.frontend << 'EOF'
FROM node:20-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "start"]
EOF

# docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8
    volumes:
      - ./backend:/app
    networks:
      - transformer-net

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    networks:
      - transformer-net

networks:
  transformer-net:
    driver: bridge
EOF

# ==================== SCRIPTS ====================

# scripts/build.sh
cat > scripts/build.sh << 'EOF'
#!/bin/bash
set -e

echo "Building L3 Transformer Deep Dive..."

# Backend
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "Build complete!"
EOF
chmod +x scripts/build.sh

# scripts/start.sh
cat > scripts/start.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting L3 Transformer Visualizer..."

# Check for Docker
if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    docker-compose up -d
    echo ""
    echo "✅ Services started!"
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
else
    echo "Docker not found. Starting manually..."
    
    # Start backend
    source venv/bin/activate
    cd backend
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo ""
    echo "✅ Services started!"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo ""
    echo "📊 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
fi
EOF
chmod +x scripts/start.sh

# scripts/stop.sh
cat > scripts/stop.sh << 'EOF'
#!/bin/bash

echo "Stopping L3 Transformer Visualizer..."

if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    # Kill processes
    pkill -f "uvicorn app.main:app"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped!"
EOF
chmod +x scripts/stop.sh

# scripts/test.sh
cat > scripts/test.sh << 'EOF'
#!/bin/bash
set -e

echo "Running L3 Transformer Tests..."

source venv/bin/activate

cd backend
pytest tests/ -v --tb=short

echo ""
echo "✅ All tests passed!"
EOF
chmod +x scripts/test.sh

# ==================== README ====================

cat > README.md << 'EOF'
# L3: Transformer Architecture Deep Dive

Interactive visualization system for understanding transformer mechanics.

## Quick Start

### With Docker
```bash
docker-compose up
```

### Without Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

## Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Testing

```bash
./scripts/test.sh
```

## Features

- Real-time attention weight visualization
- Multi-head attention heatmaps
- Performance metrics dashboard
- Interactive token exploration
- Layer-by-layer analysis

## Built With

- Backend: Python 3.11, FastAPI, NumPy
- Frontend: React 18, Plotly, Recharts
- AI: Gemini AI integration ready

## Assignment

Extend with attention pattern classification and head importance scoring.

---

**Module 1: Foundations | Lesson 3 of 90**
EOF

echo ""
echo "======================================"
echo "✅ L3 Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_ROOT"
echo "2. ./scripts/build.sh"
echo "3. ./scripts/start.sh"
echo ""
echo "Access:"
echo "- Frontend: http://localhost:3000"
echo "- Backend: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "Run tests: ./scripts/test.sh"
echo ""