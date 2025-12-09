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
