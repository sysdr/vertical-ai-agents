import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

function ModelComparison({ models }) {
  const [comparisonData, setComparisonData] = useState(null);
  const [usagePattern, setUsagePattern] = useState({
    requestsPerDay: 10000,
    avgInputTokens: 500,
    avgOutputTokens: 200,
    cacheHitRate: 0.5
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (models && Object.keys(models).length > 0) {
      fetchComparison();
    }
  }, [models, usagePattern]);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `http://localhost:8000/compare?` +
        `requests_per_day=${usagePattern.requestsPerDay}&` +
        `avg_input_tokens=${usagePattern.avgInputTokens}&` +
        `avg_output_tokens=${usagePattern.avgOutputTokens}&` +
        `cache_hit_rate=${usagePattern.cacheHitRate}`
      , { method: 'POST' });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Validate response structure
      if (data && data.comparisons && Array.isArray(data.comparisons) && data.comparisons.length > 0) {
        setComparisonData(data);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      console.error('Error fetching comparison:', error);
      setError(error.message || 'Failed to fetch comparison data. Please check if the backend is running.');
      setComparisonData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setUsagePattern(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="loader"></div>
        <p>Loading comparison data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-comparison">
        <div className="section-header">
          <h2>Model Comparison Matrix</h2>
          <p>Compare LLM parameters, costs, and performance across your usage pattern</p>
        </div>
        <div style={{
          background: '#ffebee',
          border: '1px solid #f44336',
          borderRadius: '8px',
          padding: '1.5rem',
          margin: '2rem 0',
          textAlign: 'center'
        }}>
          <h3 style={{ color: '#c62828', marginBottom: '1rem' }}>⚠️ Error Loading Comparison Data</h3>
          <p style={{ color: '#666', marginBottom: '1rem' }}>{error}</p>
          <button 
            className="calculate-btn" 
            onClick={fetchComparison}
            style={{ marginTop: '1rem' }}
          >
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  if (!comparisonData) {
    return (
      <div className="loading">
        <p>No comparison data available. Please check your connection.</p>
        <button className="calculate-btn" onClick={fetchComparison} style={{ marginTop: '1rem' }}>
          🔄 Load Data
        </button>
      </div>
    );
  }

  return (
    <div className="model-comparison">
      <div className="section-header">
        <h2>Model Comparison Matrix</h2>
        <p>Compare LLM parameters, costs, and performance across your usage pattern</p>
      </div>

      <div className="usage-config">
        <h3>Configure Usage Pattern</h3>
        <div className="config-grid">
          <div className="config-item">
            <label>Requests per Day</label>
            <input 
              type="number" 
              value={usagePattern.requestsPerDay}
              onChange={(e) => handleInputChange('requestsPerDay', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Avg Input Tokens</label>
            <input 
              type="number" 
              value={usagePattern.avgInputTokens}
              onChange={(e) => handleInputChange('avgInputTokens', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Avg Output Tokens</label>
            <input 
              type="number" 
              value={usagePattern.avgOutputTokens}
              onChange={(e) => handleInputChange('avgOutputTokens', e.target.value)}
              min="1"
            />
          </div>
          <div className="config-item">
            <label>Cache Hit Rate</label>
            <input 
              type="number" 
              step="0.1"
              value={usagePattern.cacheHitRate}
              onChange={(e) => handleInputChange('cacheHitRate', e.target.value)}
              min="0"
              max="1"
            />
          </div>
        </div>
      </div>

      <div className="recommendation-banner">
        <h3>💡 Recommended Model: {comparisonData.recommendation}</h3>
        <p>
          Best cost-efficiency for your usage pattern. 
          Monthly cost range: ${comparisonData.cost_range.min} - ${comparisonData.cost_range.max}
        </p>
      </div>

      <div className="charts-container">
        <div className="chart-box">
          <h3>Monthly Cost Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData.comparisons}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} label={{ value: 'Cost ($)', angle: -90, position: 'insideLeft' }} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
              <Bar dataKey="monthly_cost" fill="#4CAF50" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-box">
          <h3>Latency vs Cost Trade-off</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={comparisonData.comparisons}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 12 }} label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} label={{ value: 'Cost/Req ($)', angle: 90, position: 'insideRight' }} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="latency_ms" stroke="#2196F3" strokeWidth={2} />
              <Line yAxisId="right" type="monotone" dataKey="cost_per_request" stroke="#FF9800" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="comparison-table">
        <h3>Detailed Specifications</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>Context Window</th>
              <th>Monthly Cost</th>
              <th>Cost/Request</th>
              <th>Latency (ms)</th>
              <th>Max RPM</th>
              <th>Efficiency Score</th>
            </tr>
          </thead>
          <tbody>
            {comparisonData.comparisons.map((model, idx) => (
              <tr key={idx} className={model.model === comparisonData.recommendation ? 'recommended' : ''}>
                <td><strong>{model.model}</strong></td>
                <td>{model.parameters}</td>
                <td>{(model.context_window / 1000).toFixed(0)}K</td>
                <td>${model.monthly_cost}</td>
                <td>${model.cost_per_request.toFixed(6)}</td>
                <td>{model.latency_ms}</td>
                <td>{model.max_rpm}</td>
                <td>{model.cost_efficiency_score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ModelComparison;
