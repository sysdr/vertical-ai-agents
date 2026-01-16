import React, { useState, useEffect } from 'react';
import axios from 'axios';
import RerankingDashboard from './components/RerankingDashboard';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState('loading');

  useEffect(() => {
    checkHealth();
    loadMetrics();
    const interval = setInterval(loadMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      setStatus(response.data.status);
    } catch (error) {
      setStatus('error');
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎯 L22: Reranking Optimization Dashboard</h1>
        <div className={`status-badge status-${status}`}>
          {status === 'healthy' ? '✅ Operational' : '⚠️ ' + status}
        </div>
      </header>
      
      {metrics ? (
        <div className="metrics-summary">
          <div className="metric-card">
            <div className="metric-value">{metrics.total_requests || 0}</div>
            <div className="metric-label">Total Requests</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.cache_hit_rate || 0}%</div>
            <div className="metric-label">Cache Hit Rate</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.cache_hits || 0}</div>
            <div className="metric-label">Cache Hits</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.cache_misses || 0}</div>
            <div className="metric-label">Cache Misses</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.avg_latency_ms || 0}ms</div>
            <div className="metric-label">Avg Latency</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics.p95_latency_ms || 0}ms</div>
            <div className="metric-label">P95 Latency</div>
          </div>
        </div>
      ) : (
        <div className="metrics-summary">
          <div className="metric-card">
            <div className="metric-value">Loading...</div>
            <div className="metric-label">Metrics</div>
          </div>
        </div>
      )}
      
      <RerankingDashboard apiUrl={API_URL} />
    </div>
  );
}

export default App;
