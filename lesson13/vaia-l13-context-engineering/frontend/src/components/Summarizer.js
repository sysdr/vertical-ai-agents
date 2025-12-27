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
