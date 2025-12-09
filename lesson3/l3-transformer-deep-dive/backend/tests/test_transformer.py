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
