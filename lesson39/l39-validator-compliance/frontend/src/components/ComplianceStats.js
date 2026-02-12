import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './ComplianceStats.css';

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

function ComplianceStats({ stats }) {
  if (!stats) {
    return (
      <div className="stats-container">
        <h2>📊 Compliance Statistics</h2>
        <p className="loading">Loading statistics...</p>
      </div>
    );
  }

  const approvalData = [
    { name: 'Approved', value: stats.total_validations - stats.total_violations },
    { name: 'Rejected', value: stats.total_violations }
  ];

  const violationData = Object.entries(stats.violations_by_rule || {}).map(([rule, count]) => ({
    rule: rule.replace(/_/g, ' '),
    count
  }));

  return (
    <div className="stats-container">
      <h2>📊 Compliance Statistics</h2>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.total_validations}</div>
          <div className="stat-label">Total Validations</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.total_violations}</div>
          <div className="stat-label">Total Violations</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.approval_rate.toFixed(1)}%</div>
          <div className="stat-label">Approval Rate</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-value">{stats.avg_risk_score}</div>
          <div className="stat-label">Avg Risk Score</div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3>Approval Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={approvalData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {approvalData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {violationData.length > 0 && (
          <div className="chart-container">
            <h3>Violations by Rule</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={violationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="rule" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

export default ComplianceStats;
