import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [maxSubQueries, setMaxSubQueries] = useState(5);
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({
    total_queries: 0,
    successful_decompositions: 0,
    failed_decompositions: 0,
    total_sub_queries: 0,
    decompositions_with_plan: 0,
    avg_sub_queries: 0,
    avg_processing_ms: 0,
    success_rate: 0,
    status: 'unknown'
  });

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/metrics`);
      setMetrics({ ...res.data, status: 'online' });
    } catch (err) {
      setMetrics(prev => ({ ...prev, status: 'offline' }));
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  const exampleQueries = [
    "Compare Tesla's autopilot to Mercedes' system and analyze regulatory concerns",
    "What are the differences between RAG and fine-tuning for LLMs?",
    "Explain quantum computing applications in cryptography and drug discovery",
    "How do Netflix's recommendation algorithms work?"
  ];

  const handleDecompose = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setPlan(null);

    try {
      const response = await axios.post(`${API_URL}/plan`, {
        query: query,
        max_sub_queries: maxSubQueries
      });

      if (response.data.success) {
        setPlan(response.data.plan);
      } else {
        setError(response.data.error || 'Planning failed');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to decompose query');
    } finally {
      setLoading(false);
      fetchMetrics(); // Refresh metrics after each decomposition
    }
  };

  const loadExample = (exampleQuery) => {
    setQuery(exampleQuery);
    setPlan(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🧠 L36: PlannerAgent</h1>
        <p className="subtitle">Query Decomposition for Agentic RAG</p>
      </header>

      <div className="container">
        <div className="metrics-dashboard">
          <h2>📊 Metrics Dashboard</h2>
          <p className="metrics-hint">{metrics.total_queries === 0 ? 'Run a query below to see metrics update in real-time.' : 'Metrics updated with each decomposition.'}</p>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Total Queries</span>
              <span className="metric-value">{metrics.total_queries}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Successful</span>
              <span className="metric-value success">{metrics.successful_decompositions}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Failed</span>
              <span className="metric-value error">{metrics.failed_decompositions}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Total Sub-Queries</span>
              <span className="metric-value">{metrics.total_sub_queries}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Avg Sub-Queries</span>
              <span className="metric-value">{metrics.avg_sub_queries}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Avg Latency (ms)</span>
              <span className="metric-value">{metrics.avg_processing_ms}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Success Rate</span>
              <span className="metric-value">{metrics.success_rate}%</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Status</span>
              <span className={`metric-value status-${metrics.status}`}>{metrics.status}</span>
            </div>
          </div>
        </div>

        <div className="input-section">
          <h2>Enter Complex Query</h2>
          <textarea
            className="query-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a complex query to decompose into sub-queries..."
            rows="4"
          />

          <div className="controls">
            <div className="control-group">
              <label>Max Sub-Queries: {maxSubQueries}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={maxSubQueries}
                onChange={(e) => setMaxSubQueries(parseInt(e.target.value))}
              />
            </div>

            <button 
              className="decompose-btn" 
              onClick={handleDecompose}
              disabled={loading || !query.trim()}
            >
              {loading ? 'Decomposing...' : 'Decompose Query'}
            </button>
          </div>

          <div className="examples">
            <h3>Example Queries:</h3>
            {exampleQueries.map((ex, idx) => (
              <button 
                key={idx} 
                className="example-btn"
                onClick={() => loadExample(ex)}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="error-box">
            <strong>Error:</strong> {error}
          </div>
        )}

        {plan && (
          <div className="results-section">
            <div className="plan-header">
              <h2>Decomposition Plan</h2>
              <div className="plan-meta">
                <span className={`badge ${plan.estimated_complexity}`}>
                  {plan.estimated_complexity} complexity
                </span>
                <span className={`badge ${plan.strategy}`}>
                  {plan.strategy} strategy
                </span>
                <span className="badge">
                  {plan.requires_decomposition ? 'Decomposed' : 'No decomposition needed'}
                </span>
              </div>
            </div>

            <div className="original-query">
              <strong>Original Query:</strong>
              <p>{plan.original_query}</p>
            </div>

            <div className="sub-queries">
              <h3>Sub-Queries ({plan.sub_queries.length})</h3>
              {plan.sub_queries.map((sq, idx) => (
                <div key={idx} className="sub-query-card">
                  <div className="sub-query-header">
                    <span className="sub-query-number">{idx + 1}</span>
                    <span className="sub-query-text">{sq.text}</span>
                    {sq.requires_previous && (
                      <span className="dependency-badge">Depends on previous</span>
                    )}
                  </div>
                  <div className="sub-query-rationale">
                    <em>Rationale:</em> {sq.rationale}
                  </div>
                </div>
              ))}
            </div>

            <div className="execution-info">
              <h3>Execution Plan</h3>
              <p>
                {plan.strategy === 'parallel' ? (
                  <>
                    ✓ <strong>Parallel execution:</strong> All sub-queries can be executed simultaneously 
                    for optimal latency. RetrieverAgent (L37) will dispatch all searches concurrently.
                  </>
                ) : (
                  <>
                    ⚠️ <strong>Sequential execution:</strong> Sub-queries must execute in order due to 
                    dependencies. Each query waits for previous results before executing.
                  </>
                )}
              </p>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>L36 • Agentic RAG Architecture • Prepares for L37: RetrieverAgent</p>
      </footer>
    </div>
  );
}

export default App;
