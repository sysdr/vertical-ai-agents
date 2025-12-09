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
