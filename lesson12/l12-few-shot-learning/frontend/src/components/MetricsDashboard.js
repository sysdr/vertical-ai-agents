import React from 'react';

function MetricsDashboard({ metrics }) {
  if (!metrics || metrics.total_classifications === 0) {
    return (
      <div className="panel metrics-dashboard">
        <h2>Performance Metrics</h2>
        <p className="info">No classifications yet. Start classifying to see metrics!</p>
      </div>
    );
  }

  return (
    <div className="panel metrics-dashboard">
      <h2>Performance Metrics</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Total Classifications</h3>
          <div className="metric-value">{metrics.total_classifications}</div>
        </div>

        <div className="metric-card">
          <h3>Avg Latency</h3>
          <div className="metric-value">{metrics.avg_latency_ms?.toFixed(0)}ms</div>
        </div>

        <div className="metric-card">
          <h3>Avg Tokens</h3>
          <div className="metric-value">{metrics.avg_tokens?.toFixed(0)}</div>
        </div>

        <div className="metric-card">
          <h3>Avg Confidence</h3>
          <div className="metric-value">{(metrics.avg_confidence * 100)?.toFixed(1)}%</div>
        </div>
      </div>

      {metrics.recent_classifications && (
        <div className="recent-activity">
          <h3>Recent Classifications</h3>
          <div className="activity-list">
            {metrics.recent_classifications.slice().reverse().map((item, idx) => (
              <div key={idx} className="activity-item">
                <div className="activity-query">{item.query}</div>
                <div className="activity-details">
                  <span className="classification-badge">{item.classification}</span>
                  <span>{item.num_examples} examples</span>
                  <span>{item.latency_ms.toFixed(0)}ms</span>
                  <span>{(item.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default MetricsDashboard;
