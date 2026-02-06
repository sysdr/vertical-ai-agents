import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function MetricsDashboard({ metrics, summary }) {
  const chartData = metrics.map((m, idx) => ({
    name: `Req ${idx + 1}`,
    iterations: m.total_iterations || 0,
    tokens: (m.total_tokens || 0) / 1000, // Scale to thousands
    error: m.error ? 1 : 0
  }));

  return (
    <div className="dashboard">
      <h2>📈 Execution Metrics Dashboard</h2>
      
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">📊</div>
          <div className="card-content">
            <div className="card-value">{summary.total_executions || 0}</div>
            <div className="card-label">Total Executions</div>
          </div>
        </div>
        
        <div className="summary-card">
          <div className="card-icon">🔄</div>
          <div className="card-content">
            <div className="card-value">{(summary.average_iterations || 0).toFixed(1)}</div>
            <div className="card-label">Avg Iterations</div>
          </div>
        </div>
        
        <div className="summary-card">
          <div className="card-icon">🎫</div>
          <div className="card-content">
            <div className="card-value">{((summary.average_tokens || 0) / 1000).toFixed(1)}K</div>
            <div className="card-label">Avg Tokens</div>
          </div>
        </div>
        
        <div className="summary-card error">
          <div className="card-icon">⚠️</div>
          <div className="card-content">
            <div className="card-value">{summary.budget_violations || 0}</div>
            <div className="card-label">Budget Violations</div>
          </div>
        </div>
      </div>

      {chartData.length > 0 && (
        <div className="chart-container">
          <h3>Recent Execution Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="iterations" stroke="#4CAF50" name="Iterations" />
              <Line yAxisId="right" type="monotone" dataKey="tokens" stroke="#2196F3" name="Tokens (K)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="recent-executions">
        <h3>Recent Executions</h3>
        {metrics.length === 0 ? (
          <p className="no-data">No executions yet. Try running a task above!</p>
        ) : (
          <div className="executions-list">
            {metrics.slice().reverse().map((m, idx) => (
              <div key={idx} className={`execution-item ${m.error ? 'has-error' : ''}`}>
                <div className="execution-header">
                  <span className="task-preview">{m.task}</span>
                  <span className="timestamp">{new Date(m.timestamp).toLocaleTimeString()}</span>
                </div>
                <div className="execution-metrics">
                  <span>🔄 {m.total_iterations || 0} iterations</span>
                  <span>🎫 {(m.total_tokens || 0).toLocaleString()} tokens</span>
                  <span>🌍 {m.environment}</span>
                  {m.error && <span className="error-badge">⚠️ {m.error}</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricsDashboard;
