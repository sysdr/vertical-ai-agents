import React, { useState } from 'react';
import axios from 'axios';

function QueryInterface({ apiUrl }) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('hybrid');
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${apiUrl}/api/query`, {
        query: query,
        mode: mode,
        top_k: topK,
        enable_reranking: false
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  return (
    <div className="query-container">
      <div className="query-header">
        <h2>Query RAG System</h2>
        <p>Ask questions and explore LangChain-powered modular retrieval</p>
      </div>

      <div className="query-input-section">
        <div className="query-input-group">
          <input
            type="text"
            className="query-input"
            placeholder="Enter your question (e.g., What is retrieval augmented generation?)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <button
            className="query-btn"
            onClick={handleQuery}
            disabled={loading || !query.trim()}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        <div className="options-row">
          <div className="option-group">
            <label>Mode:</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="hybrid">Hybrid Search</option>
              <option value="vector">Vector Only</option>
            </select>
          </div>

          <div className="option-group">
            <label>Top-K:</label>
            <input
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
          </div>
        </div>
      </div>

      {error && (
        <div className="error-box">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="results-section">
          <div className="answer-box">
            <h3>💡 Answer</h3>
            <div className="answer-text">{result.answer}</div>
          </div>

          <div className="metrics-box">
            <div className="metric-card">
              <h4>Retrieval Time</h4>
              <div className="value">{(result.retrieval_time * 1000).toFixed(0)}ms</div>
            </div>
            <div className="metric-card">
              <h4>Generation Time</h4>
              <div className="value">{(result.generation_time * 1000).toFixed(0)}ms</div>
            </div>
            <div className="metric-card">
              <h4>Total Time</h4>
              <div className="value">
                {((result.retrieval_time + result.generation_time) * 1000).toFixed(0)}ms
              </div>
            </div>
            <div className="metric-card">
              <h4>Sources Found</h4>
              <div className="value">{result.sources.length}</div>
            </div>
          </div>

          {result.chain_trace && (
            <div className="chain-trace">
              <h4>⛓️ Chain Execution Trace</h4>
              {result.chain_trace.map((step, idx) => (
                <div key={idx} className="trace-step">
                  <span className="trace-label">{step.step}</span>
                  <span className="trace-value">
                    {step.duration ? `${(step.duration * 1000).toFixed(0)}ms` : 
                     step.docs_found ? `${step.docs_found} docs` :
                     step.tokens ? `${step.tokens} tokens` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="sources-box">
            <h3>📚 Retrieved Sources</h3>
            {result.sources.map((source, idx) => (
              <div key={idx} className="source-item">
                <div className="source-header">
                  <span className="source-number">Source {idx + 1}</span>
                  <span className="source-score">
                    Score: {(source.relevance_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="source-content">{source.content}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default QueryInterface;
