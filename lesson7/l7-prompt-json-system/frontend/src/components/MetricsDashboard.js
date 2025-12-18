import React from 'react';
import './MetricsDashboard.css';

function MetricsDashboard({ metrics }) {
  if (!metrics) {
    return (
      <div className="metrics-dashboard">
        <h2 className="section-title">Live Metrics</h2>
        <p className="loading-message">Loading metrics...</p>
      </div>
    );
  }

  const strategyData = [
    { name: 'Direct Parse', count: metrics.strategy_counts.direct, color: '#28a745' },
    { name: 'Regex Extract', count: metrics.strategy_counts.regex, color: '#ffc107' },
    { name: 'LLM Repair', count: metrics.strategy_counts.repair, color: '#17a2b8' },
    { name: 'Failed', count: metrics.strategy_counts.failed, color: '#dc3545' }
  ];

  const maxCount = Math.max(...strategyData.map(d => d.count), 1);

  return (
    <div className="metrics-dashboard">
      <h2 className="section-title">Live Metrics</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{metrics.total_requests}</div>
          <div className="metric-label">Total Requests</div>
        </div>

        <div className="metric-card success">
          <div className="metric-value">{metrics.successful_parses}</div>
          <div className="metric-label">Successful</div>
        </div>

        <div className="metric-card error">
          <div className="metric-value">{metrics.failed_parses}</div>
          <div className="metric-label">Failed</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.success_rate.toFixed(1)}%</div>
          <div className="metric-label">Success Rate</div>
        </div>
      </div>

      <div className="chart-section">
        <h3>Parse Strategy Distribution</h3>
        <div className="bar-chart">
          {strategyData.map((strategy, i) => (
            <div key={i} className="bar-row">
              <div className="bar-label">{strategy.name}</div>
              <div className="bar-container">
                <div
                  className="bar-fill"
                  style={{
                    width: `${(strategy.count / maxCount) * 100}%`,
                    background: strategy.color
                  }}
                >
                  <span className="bar-count">{strategy.count}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="performance-section">
        <h3>Performance</h3>
        <div className="performance-stat">
          <span className="stat-label">Average Parse Time:</span>
          <span className="stat-value">{metrics.avg_parse_time_ms.toFixed(2)} ms</span>
        </div>
        <div className="performance-stat">
          <span className="stat-label">Total Parse Operations:</span>
          <span className="stat-value">{metrics.parse_times.length}</span>
        </div>
      </div>

      <div className="insights-section">
        <h3>Insights</h3>
        <div className="insight-item">
          {metrics.strategy_counts.direct / metrics.total_requests > 0.9 ? (
            <span className="insight-good">✅ Excellent! 90%+ direct parse success</span>
          ) : metrics.strategy_counts.direct / metrics.total_requests > 0.7 ? (
            <span className="insight-warning">⚠️ 70-90% direct parse - room for improvement</span>
          ) : (
            <span className="insight-bad">❌ Low direct parse rate - check prompts</span>
          )}
        </div>
        
        {metrics.strategy_counts.repair > 0 && (
          <div className="insight-item">
            <span className="insight-warning">
              🔧 {metrics.strategy_counts.repair} responses required LLM repair
            </span>
          </div>
        )}
        
        {metrics.avg_parse_time_ms > 100 && (
          <div className="insight-item">
            <span className="insight-warning">
              ⚡ Parse time above 100ms - consider optimization
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricsDashboard;
