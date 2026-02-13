import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [simulateFailure, setSimulateFailure] = useState(false);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_URL}/metrics`);
      setMetrics(res.data);
    } catch (error) {
      console.error('Metrics fetch error:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await axios.post(`${API_URL}/query`, {
        query,
        simulate_failure: simulateFailure
      });
      setResponse(res.data);
    } catch (error) {
      setResponse({
        success: false,
        error: error.response?.data?.detail || 'Request failed'
      });
    } finally {
      setLoading(false);
      fetchMetrics();
    }
  };

  const getStateColor = (state) => {
    switch(state) {
      case 'closed': return '#10b981';
      case 'half_open': return '#f59e0b';
      case 'open': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const DEMO_QUERIES = [
    { label: 'Fallback logic (success)', query: 'What is fallback logic in software systems?', simulateFailure: false },
    { label: 'Circuit breaker (success)', query: 'Explain the circuit breaker pattern for resilience', simulateFailure: false },
    { label: 'Exponential backoff (success)', query: 'How does exponential backoff work in retry logic?', simulateFailure: false },
    { label: 'Query reformulation (success)', query: 'What strategies are used for query reformulation?', simulateFailure: false },
    { label: 'Graceful degradation (success)', query: 'Describe graceful degradation in distributed systems', simulateFailure: false },
    { label: 'Validator feedback (success)', query: 'How does validator feedback improve planner output?', simulateFailure: false },
    { label: 'Retry mechanism (simulated failure)', query: 'Test retry mechanism', simulateFailure: true },
    { label: 'Validation retry (simulated failure)', query: 'Demo validation retries', simulateFailure: true },
    { label: 'Compliance check (simulated failure)', query: 'Test compliance validation', simulateFailure: true }
  ];

  const runDemoQuery = async (demo) => {
    setQuery(demo.query);
    setSimulateFailure(demo.simulateFailure);
    setResponse(null);
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/query`, {
        query: demo.query,
        simulate_failure: demo.simulateFailure
      });
      setResponse(res.data);
    } catch (error) {
      setResponse({
        success: false,
        error: error.response?.data?.detail || 'Request failed'
      });
    } finally {
      setLoading(false);
      fetchMetrics();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔄 L40: Fallback Logic & Self-Healing</h1>
        <p>Validator ↔ Planner Feedback Loops with Circuit Breaker</p>
      </header>

      <div className="container">
        {/* Dashboard Metrics */}
        {metrics && (
          <div className="status-panel">
            <h2>Dashboard Metrics</h2>
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">Circuit State:</span>
                <span 
                  className="status-badge"
                  style={{ backgroundColor: getStateColor(metrics.circuit_breaker?.state) }}
                >
                  {(metrics.circuit_breaker?.state || 'closed').toUpperCase()}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Total Queries:</span>
                <span className="status-value">{metrics.total_queries ?? 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Successful:</span>
                <span className="status-value" style={{ color: '#10b981' }}>{metrics.successful_queries ?? 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Failed:</span>
                <span className="status-value" style={{ color: '#ef4444' }}>{metrics.failed_queries ?? 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Retry Events:</span>
                <span className="status-value">{metrics.retry_history_count ?? 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Circuit Failures:</span>
                <span className="status-value">{metrics.circuit_breaker?.failure_count ?? 0}</span>
              </div>
            </div>
          </div>
        )}

        {/* Demo Queries */}
        <div className="query-panel">
          <h2>Demo Queries</h2>
          <p className="demo-hint">Click a query to select and execute it:</p>
          <div className="demo-buttons">
            {DEMO_QUERIES.map((demo, idx) => (
              <button
                key={idx}
                type="button"
                className="demo-btn"
                onClick={() => runDemoQuery(demo)}
                disabled={loading}
              >
                {demo.simulateFailure ? '⚠️ ' : '✓ '}{demo.label}
              </button>
            ))}
          </div>
        </div>

        {/* Query Form */}
        <div className="query-panel">
          <h2>Test Fallback System</h2>
          <form onSubmit={handleSubmit}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query..."
              rows="4"
              disabled={loading}
            />
            <div className="controls">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={simulateFailure}
                  onChange={(e) => setSimulateFailure(e.target.checked)}
                />
                Simulate Validation Failure
              </label>
              <button type="submit" disabled={loading}>
                {loading ? '⏳ Processing...' : '🚀 Execute Query'}
              </button>
            </div>
          </form>
        </div>

        {/* Response Panel */}
        {response && (
          <div className={`response-panel ${response.success ? 'success' : 'error'}`}>
            <h2>{response.success ? '✅ Success' : '❌ Failed'}</h2>
            <div className="response-details">
              <div className="detail-row">
                <strong>Retry Count:</strong> {response.retry_count}
              </div>
              <div className="detail-row">
                <strong>Latency:</strong> {response.total_latency_ms?.toFixed(2)}ms
              </div>
              {response.metadata && (
                <div className="detail-row">
                  <strong>Attempts:</strong> {response.metadata.attempts}
                  {response.metadata.reformulations > 0 && (
                    <span className="badge">
                      {response.metadata.reformulations} reformulation(s)
                    </span>
                  )}
                </div>
              )}
              {response.error && (
                <div className="error-message">
                  <strong>Error:</strong> {response.error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Retry History */}
        {metrics && metrics.recent_retries && metrics.recent_retries.length > 0 && (
          <div className="history-panel">
            <h2>Recent Retry History</h2>
            <div className="retry-list">
              {metrics.recent_retries.slice().reverse().map((retry, idx) => (
                <div key={idx} className="retry-item">
                  <div className="retry-header">
                    <span className="retry-attempt">Attempt {retry.attempt_number + 1}</span>
                    <span className={`retry-status ${retry.validation_result?.success ? 'success' : 'failed'}`}>
                      {retry.validation_result?.success ? '✓' : '✗'}
                    </span>
                  </div>
                  <div className="retry-details">
                    {retry.delay_seconds > 0 && (
                      <span>Delay: {retry.delay_seconds.toFixed(2)}s</span>
                    )}
                    {retry.reformulation_applied && (
                      <span className="badge">Reformulated</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
