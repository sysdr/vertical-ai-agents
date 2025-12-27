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
