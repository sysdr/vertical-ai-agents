import React from 'react';
import { Activity, TrendingDown, DollarSign, Zap } from 'lucide-react';

function Dashboard({ stats }) {
  return (
    <div className="dashboard">
      <div className="stat-card">
        <Activity className="stat-icon" />
        <div className="stat-content">
          <span className="stat-label">Total Optimizations</span>
          <span className="stat-value">{stats.totalOptimizations}</span>
        </div>
      </div>

      <div className="stat-card">
        <TrendingDown className="stat-icon success" />
        <div className="stat-content">
          <span className="stat-label">Tokens Saved</span>
          <span className="stat-value">{stats.totalTokensSaved.toLocaleString()}</span>
        </div>
      </div>

      <div className="stat-card">
        <Zap className="stat-icon warning" />
        <div className="stat-content">
          <span className="stat-label">Avg Compression</span>
          <span className="stat-value">{stats.averageCompressionRatio.toFixed(2)}x</span>
        </div>
      </div>

      <div className="stat-card">
        <DollarSign className="stat-icon" />
        <div className="stat-content">
          <span className="stat-label">Cost Savings</span>
          <span className="stat-value">${stats.totalCostSavings.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
