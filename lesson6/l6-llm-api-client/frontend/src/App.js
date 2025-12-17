import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/stats`);
      setStats(res.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await axios.post(`${API_BASE}/api/generate`, {
        prompt: prompt,
        temperature: 0.7,
        max_tokens: 1000
      });

      setResponse(res.data);
      
      if (res.data.success) {
        setHistory(prev => [{
          prompt: prompt.substring(0, 50) + '...',
          tokens: res.data.output_tokens,
          cost: res.data.cost_usd,
          latency: res.data.latency_ms,
          timestamp: new Date().toISOString()
        }, ...prev].slice(0, 10));
      }
    } catch (error) {
      setResponse({
        success: false,
        error: 'network_error',
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_BASE}/api/stats/reset`);
      fetchStats();
      setHistory([]);
    } catch (error) {
      console.error('Error resetting stats:', error);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🚀 L6: LLM API Client</h1>
        <p>Production-Grade API Integration with Rate Limiting & Circuit Breaker</p>
      </header>

      <div className="container">
        {/* Stats Dashboard */}
        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <h3>Rate Limiter</h3>
              <div className="stat-value">{stats.rate_limiter.rpm_available}</div>
              <div className="stat-label">RPM Available</div>
              <div className="stat-subvalue">
                {Math.round(stats.rate_limiter.utilization_rpm * 100)}% utilized
              </div>
            </div>

            <div className="stat-card">
              <h3>Circuit Breaker</h3>
              <div className={`stat-value status-${stats.circuit_breaker.state}`}>
                {stats.circuit_breaker.state.toUpperCase()}
              </div>
              <div className="stat-label">Status</div>
              <div className="stat-subvalue">
                {stats.circuit_breaker.failure_count} failures
              </div>
            </div>

            <div className="stat-card">
              <h3>Cost Tracking</h3>
              <div className="stat-value">${stats.cost_tracker.total_cost_usd}</div>
              <div className="stat-label">Total Cost</div>
              <div className="stat-subvalue">
                {stats.cost_tracker.total_requests} requests
              </div>
            </div>

            <div className="stat-card">
              <h3>Performance</h3>
              <div className="stat-value">
                ${(stats.cost_tracker.avg_cost_per_request * 1000).toFixed(3)}
              </div>
              <div className="stat-label">Avg Cost ($/1K req)</div>
              <div className="stat-subvalue">
                {stats.cost_tracker.requests_per_hour} req/hr
              </div>
            </div>
          </div>
        )}

        {/* Prompt Interface */}
        <div className="prompt-section">
          <h2>Generate Content</h2>
          <textarea
            className="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt here... (try: 'Write a haiku about AI')"
            rows={4}
          />
          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
            >
              {loading ? '⏳ Generating...' : '✨ Generate'}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              🔄 Reset Stats
            </button>
          </div>
        </div>

        {/* Response Display */}
        {response && (
          <div className={`response-section ${response.success ? 'success' : 'error'}`}>
            <h2>{response.success ? '✅ Response' : '❌ Error'}</h2>
            
            {response.success ? (
              <>
                <div className="response-text">{response.text}</div>
                <div className="response-meta">
                  <span>📊 {response.input_tokens} → {response.output_tokens} tokens</span>
                  <span>💰 ${response.cost_usd.toFixed(6)}</span>
                  <span>⚡ {response.latency_ms}ms</span>
                  <span>🔖 {response.request_id}</span>
                </div>
              </>
            ) : (
              <div className="error-message">
                <strong>{response.error}:</strong> {response.message}
                {response.wait_seconds && (
                  <p>⏰ Rate limited. Retry in {response.wait_seconds.toFixed(1)}s</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Request History */}
        {history.length > 0 && (
          <div className="history-section">
            <h2>📜 Recent Requests</h2>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                  <th>Latency</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item, idx) => (
                  <tr key={idx}>
                    <td>{item.prompt}</td>
                    <td>{item.tokens}</td>
                    <td>${item.cost.toFixed(6)}</td>
                    <td>{item.latency}ms</td>
                    <td>{new Date(item.timestamp).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>L6: Interacting with LLM APIs | VAIA Curriculum</p>
        <p>Built with FastAPI + React | Powered by Gemini AI</p>
      </footer>
    </div>
  );
}

export default App;
