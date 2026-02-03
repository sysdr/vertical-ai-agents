import React from 'react';
import './MetricsDashboard.css';

function MetricsDashboard({ stats }) {
  if (!stats) {
    return (
      <div className="metrics-dashboard">
        <div className="loading">Connecting to metrics...</div>
      </div>
    );
  }

  const hasData = stats.total_plans > 0;
  const formatValue = (val, formatter) => {
    if (!hasData && (val === 0 || val === undefined)) return '—';
    return formatter ? formatter(val) : String(val);
  };

  return (
    <div className="metrics-dashboard">
      <h2>Planning Metrics</h2>
      {!hasData && (
        <div className="no-data-banner">
          <p>Run a planning task below to see real-time metrics.</p>
        </div>
      )}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📋</div>
          <div className="stat-content">
            <span className="stat-label">Total Plans</span>
            <span className="stat-value">{formatValue(stats.total_plans, v => v.toLocaleString())}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🔄</div>
          <div className="stat-content">
            <span className="stat-label">Total Iterations</span>
            <span className="stat-value">{formatValue(stats.total_iterations, v => v.toLocaleString())}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-content">
            <span className="stat-label">Total Tokens</span>
            <span className="stat-value">{formatValue(stats.total_tokens, v => v.toLocaleString())}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <span className="stat-label">Avg Latency</span>
            <span className="stat-value">{formatValue(stats.avg_latency_ms, v => v.toFixed(2) + 'ms')}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <span className="stat-label">Avg Confidence</span>
            <span className="stat-value">{formatValue(stats.avg_confidence, v => (v * 100).toFixed(1) + '%')}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">⏱️</div>
          <div className="stat-content">
            <span className="stat-label">Total Latency</span>
            <span className="stat-value">{formatValue(stats.total_latency_ms, v => v.toFixed(2) + 'ms')}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MetricsDashboard;
