import React from 'react';

function StatsPanel({ stats }) {
  return (
    <div className="panel stats-panel">
      <h2>📊 Collection Stats</h2>
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats.total_documents || 0}</h3>
          <p>Indexed Documents</p>
        </div>
        <div className="stat-card">
          <h3>{stats.index_type || 'HNSW'}</h3>
          <p>Index Algorithm</p>
        </div>
        <div className="stat-card">
          <h3>{stats.distance_metric || 'Cosine'}</h3>
          <p>Distance Metric</p>
        </div>
      </div>
    </div>
  );
}

export default StatsPanel;
