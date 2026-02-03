import React from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './MetricsDashboard.css';

function MetricsDashboard({ stats }) {
  if (!stats) {
    return (
      <div className="metrics-dashboard">
        <div className="loading">Connecting to metrics stream...</div>
      </div>
    );
  }

  const hasData = stats.total_requests > 0;
  const latencyData = hasData ? [
    { name: 'P50', value: stats.p50_latency_ms },
    { name: 'P95', value: stats.p95_latency_ms },
    { name: 'P99', value: stats.p99_latency_ms },
  ] : [];

  const formatValue = (val, formatter) => {
    if (!hasData) return '—';
    return formatter ? formatter(val) : String(val);
  };

  return (
    <div className="metrics-dashboard">
      <h2>Performance Metrics</h2>
      
      {!hasData && (
        <div className="no-data-banner">
          <p>📋 No metrics yet. Run a query in the Query Interface to see real-time observability data.</p>
        </div>
      )}
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <span className="stat-label">Total Requests</span>
            <span className="stat-value">{formatValue(stats.total_requests, v => v.toLocaleString())}</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">⚡</div>
          <div className="stat-content">
            <span className="stat-label">Avg Latency</span>
            <span className="stat-value">{formatValue(stats.avg_latency_ms, v => `${v.toFixed(2)}ms`)}</span>
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
          <div className="stat-icon">💰</div>
          <div className="stat-content">
            <span className="stat-label">Total Cost</span>
            <span className="stat-value">{formatValue(stats.total_cost_usd, v => '$' + v.toFixed(4))}</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">❌</div>
          <div className="stat-content">
            <span className="stat-label">Error Rate</span>
            <span className="stat-value">{formatValue(stats.error_rate, v => `${(v * 100).toFixed(2)}%`)}</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">💾</div>
          <div className="stat-content">
            <span className="stat-label">Cache Hit Rate</span>
            <span className="stat-value">{formatValue(stats.cache_hit_rate, v => `${(v * 100).toFixed(2)}%`)}</span>
          </div>
        </div>
      </div>

      <div className="chart-section">
        <h3>Latency Percentiles</h3>
        {hasData ? (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={latencyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="name" stroke="#718096" />
            <YAxis stroke="#718096" />
            <Tooltip 
              contentStyle={{ background: '#fff', border: '1px solid #e2e8f0' }}
              formatter={(value) => `${value.toFixed(2)}ms`}
            />
            <Bar dataKey="value" fill="#667eea" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        ) : (
          <div className="chart-placeholder">Run a query to populate latency percentiles</div>
        )}
      </div>
    </div>
  );
}

export default MetricsDashboard;
