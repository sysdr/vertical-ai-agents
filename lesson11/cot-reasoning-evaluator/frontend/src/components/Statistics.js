import React from 'react';

function Statistics({ stats }) {
  if (!stats.total_traces) {
    return <p className="empty-state">Submit queries to see statistics</p>;
  }

  return (
    <div className="statistics">
      <div className="stat-card">
        <div className="stat-value">{stats.total_traces}</div>
        <div className="stat-label">Total Traces</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{(stats.avg_quality * 100).toFixed(0)}%</div>
        <div className="stat-label">Avg Quality</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{stats.high_quality_count}</div>
        <div className="stat-label">High Quality</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{(stats.max_quality * 100).toFixed(0)}%</div>
        <div className="stat-label">Best Score</div>
      </div>
    </div>
  );
}

export default Statistics;
