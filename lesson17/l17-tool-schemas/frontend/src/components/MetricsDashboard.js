import React, { useState, useEffect } from 'react';

function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/metrics');
      const data = await response.json();
      setMetrics(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load metrics:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="metrics-dashboard">Loading metrics...</div>;
  }

  if (!metrics) {
    return <div className="metrics-dashboard">Failed to load metrics</div>;
  }

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  return (
    <div className="metrics-dashboard">
      <div className="section-header">
        <h2>Metrics Dashboard</h2>
        <p>Real-time statistics and tool usage metrics</p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Total Queries</h3>
          <div className="metric-value">{metrics.total_queries || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Total Tool Calls</h3>
          <div className="metric-value">{metrics.total_tool_calls || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Total Validations</h3>
          <div className="metric-value">{metrics.total_validations || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Validation Success Rate</h3>
          <div className="metric-value">{metrics.success_rate?.toFixed(1) || 0}%</div>
        </div>

        <div className="metric-card">
          <h3>Query Success</h3>
          <div className="metric-value success">{metrics.query_success || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Query Failures</h3>
          <div className="metric-value error">{metrics.query_failures || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Validation Success</h3>
          <div className="metric-value success">{metrics.validation_success || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Validation Failures</h3>
          <div className="metric-value error">{metrics.validation_failures || 0}</div>
        </div>

        <div className="metric-card">
          <h3>Uptime</h3>
          <div className="metric-value">{formatUptime(metrics.uptime_seconds || 0)}</div>
        </div>
      </div>

      <div className="tool-usage-section">
        <h3>Tool Usage Statistics</h3>
        {Object.keys(metrics.tool_usage || {}).length > 0 ? (
          <div className="tool-usage-list">
            {Object.entries(metrics.tool_usage).map(([tool, count]) => (
              <div key={tool} className="tool-usage-item">
                <span className="tool-name">{tool}</span>
                <span className="tool-count">{count}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="no-data">No tool calls yet. Execute queries to see usage statistics.</p>
        )}
      </div>
    </div>
  );
}

export default MetricsDashboard;
