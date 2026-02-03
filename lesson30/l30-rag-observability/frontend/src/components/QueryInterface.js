import React, { useState } from 'react';
import axios from 'axios';
import './QueryInterface.css';

function QueryInterface() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const result = await axios.post('http://localhost:8000/api/query', {
        query: query
      });
      setResponse(result.data);
    } catch (error) {
      console.error('Query error:', error);
      setResponse({ error: 'Failed to process query' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-interface">
      <h2>Query Interface</h2>
      
      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your question..."
          rows={4}
          disabled={loading}
        />
        
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Processing...' : 'Submit Query'}
        </button>
      </form>

      {response && (
        <div className="response-section">
          <h3>Response</h3>
          <div className="response-content">
            {response.error ? (
              <p className="error">{response.error}</p>
            ) : (
              <>
                <p className="answer">{response.response}</p>
                
                <div className="metrics-grid">
                  <div className="metric-card">
                    <span className="metric-label">Total Latency</span>
                    <span className="metric-value">
                      {response.metrics.total_latency_ms.toFixed(2)}ms
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Embedding</span>
                    <span className="metric-value">
                      {response.metrics.embedding_latency_ms.toFixed(2)}ms
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Retrieval</span>
                    <span className="metric-value">
                      {response.metrics.retrieval_latency_ms.toFixed(2)}ms
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Generation</span>
                    <span className="metric-value">
                      {response.metrics.generation_latency_ms.toFixed(2)}ms
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Chunks Retrieved</span>
                    <span className="metric-value">{response.metrics.chunks_retrieved}</span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Avg Similarity</span>
                    <span className="metric-value">
                      {response.metrics.avg_similarity.toFixed(3)}
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Cost</span>
                    <span className="metric-value">
                      ${response.metrics.cost_usd.toFixed(6)}
                    </span>
                  </div>
                  
                  <div className="metric-card">
                    <span className="metric-label">Trace ID</span>
                    <span className="metric-value trace-id">
                      {response.trace_id.substring(0, 8)}...
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default QueryInterface;
