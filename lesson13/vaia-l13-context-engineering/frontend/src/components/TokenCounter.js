import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, CheckCircle } from 'lucide-react';

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
