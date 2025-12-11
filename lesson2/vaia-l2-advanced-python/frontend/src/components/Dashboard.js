import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const API_BASE = 'http://localhost:8000';

function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [agentId, setAgentId] = useState('agent-12345678');
  const [prompts, setPrompts] = useState('Explain async/await in Python\nWhat are decorators?\nHow does Pydantic work?');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await axios.get(`${API_BASE}/metrics`);
        setMetrics(res.data);
      } catch (err) {
        console.error('Metrics fetch error:', err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const promptList = prompts.split('\n').filter(p => p.trim());
      const res = await axios.post(`${API_BASE}/process`, {
        agent_id: agentId,
        prompts: promptList,
        temperature: parseFloat(temperature),
        max_tokens: parseInt(maxTokens)
      });
      setResponse(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>🚀 VAIA L2: Advanced Python Patterns</h1>
        <p>Async Processing • Decorators • Pydantic Validation</p>
      </header>

      {metrics && (
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{metrics.total_requests}</div>
            <div className="metric-label">Total Requests</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.active_tasks}</div>
            <div className="metric-label">Active Tasks</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.avg_latency_ms}ms</div>
            <div className="metric-label">Avg Latency</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(metrics.success_rate * 100).toFixed(1)}%</div>
            <div className="metric-label">Success Rate</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{(metrics.cache_hit_rate * 100).toFixed(1)}%</div>
            <div className="metric-label">Cache Hit Rate</div>
          </div>
        </div>
      )}

      <div className="content-grid">
        <div className="input-section">
          <h2>Agent Request</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Agent ID</label>
              <input
                type="text"
                value={agentId}
                onChange={(e) => setAgentId(e.target.value)}
                pattern="^agent-[a-z0-9]{8}$"
                placeholder="agent-12345678"
              />
            </div>

            <div className="form-group">
              <label>Prompts (one per line)</label>
              <textarea
                value={prompts}
                onChange={(e) => setPrompts(e.target.value)}
                rows="6"
                placeholder="Enter prompts..."
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                  min="1"
                  max="8192"
                />
              </div>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? '⏳ Processing...' : '▶️ Process Async'}
            </button>
          </form>
        </div>

        <div className="output-section">
          <h2>Response</h2>
          
          {error && (
            <div className="error-box">
              <strong>❌ Error:</strong> {error}
            </div>
          )}

          {response && (
            <div className="response-box">
              <div className="response-header">
                <span className={response.cached ? 'badge cached' : 'badge fresh'}>
                  {response.cached ? '💾 Cached' : '🔥 Fresh'}
                </span>
                <span className="latency">{response.latency_ms}ms</span>
              </div>
              
              <div className="results">
                {response.results.map((result, idx) => (
                  <div key={idx} className="result-item">
                    <div className="result-header">Result {idx + 1}</div>
                    <div className="result-text">{result}</div>
                  </div>
                ))}
              </div>
              
              <div className="response-footer">
                <small>Agent: {response.agent_id} • {response.timestamp}</small>
              </div>
            </div>
          )}

          {!response && !error && !loading && (
            <div className="placeholder">
              <p>✨ Submit a request to see async processing in action</p>
              <ul>
                <li>Multiple prompts processed concurrently</li>
                <li>Automatic retries with exponential backoff</li>
                <li>Response caching for efficiency</li>
                <li>Type-safe validation with Pydantic</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      <footer className="footer">
        <p>Building on L1's foundation → Preparing for L3's Transformer architecture</p>
      </footer>
    </div>
  );
}

export default Dashboard;
