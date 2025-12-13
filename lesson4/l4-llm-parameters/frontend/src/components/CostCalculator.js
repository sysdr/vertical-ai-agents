import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

function CostCalculator({ models }) {
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash-exp');
  const [requestsPerDay, setRequestsPerDay] = useState(10000);
  const [avgInputTokens, setAvgInputTokens] = useState(500);
  const [avgOutputTokens, setAvgOutputTokens] = useState(200);
  const [cacheHitRate, setCacheHitRate] = useState(0.5);
  const [costData, setCostData] = useState(null);
  const [loading, setLoading] = useState(false);

  const calculateCost = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analyze/cost', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: selectedModel,
          requests_per_day: requestsPerDay,
          avg_input_tokens: avgInputTokens,
          avg_output_tokens: avgOutputTokens,
          cache_hit_rate: cacheHitRate
        })
      });
      const data = await response.json();
      setCostData(data);
    } catch (error) {
      console.error('Error calculating cost:', error);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#4CAF50', '#FF9800', '#2196F3'];

  const pieData = costData ? [
    { name: 'Cached Input', value: costData.breakdown.cached_input_cost_per_request * requestsPerDay * 30 },
    { name: 'Uncached Input', value: costData.breakdown.uncached_input_cost_per_request * requestsPerDay * 30 },
    { name: 'Output', value: costData.breakdown.output_cost_per_request * requestsPerDay * 30 }
  ] : [];

  return (
    <div className="cost-calculator">
      <div className="section-header">
        <h2>Cost Calculator</h2>
        <p>Project monthly costs with caching optimization</p>
      </div>

      <div className="calculator-form">
        <div className="form-row">
          <div className="form-group">
            <label>Select Model</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {Object.keys(models).map(modelName => (
                <option key={modelName} value={modelName}>{modelName}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Requests per Day</label>
            <input 
              type="number" 
              value={requestsPerDay}
              onChange={(e) => setRequestsPerDay(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Average Input Tokens</label>
            <input 
              type="number" 
              value={avgInputTokens}
              onChange={(e) => setAvgInputTokens(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>Average Output Tokens</label>
            <input 
              type="number" 
              value={avgOutputTokens}
              onChange={(e) => setAvgOutputTokens(parseInt(e.target.value) || 0)}
              min="1"
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Cache Hit Rate (0.0 - 1.0)</label>
            <input 
              type="number" 
              step="0.1"
              value={cacheHitRate}
              onChange={(e) => setCacheHitRate(parseFloat(e.target.value) || 0)}
              min="0"
              max="1"
            />
          </div>
          <div className="form-group">
            <button className="calculate-btn" onClick={calculateCost} disabled={loading}>
              {loading ? 'Calculating...' : '💰 Calculate Cost'}
            </button>
          </div>
        </div>
      </div>

      {costData && (
        <div className="cost-results">
          <div className="cost-cards">
            <div className="cost-card primary">
              <h3>Monthly Cost</h3>
              <div className="cost-value">${costData.monthly_cost.toFixed(2)}</div>
              <p>{costData.total_requests_monthly.toLocaleString()} requests</p>
            </div>
            <div className="cost-card">
              <h3>Per Request</h3>
              <div className="cost-value">${costData.cost_per_request.toFixed(6)}</div>
              <p>Average cost</p>
            </div>
            <div className="cost-card">
              <h3>Yearly Projection</h3>
              <div className="cost-value">${costData.yearly_cost.toFixed(2)}</div>
              <p>Annual spend</p>
            </div>
            <div className="cost-card success">
              <h3>Cache Savings</h3>
              <div className="cost-value">${costData.cache_savings_monthly.toFixed(2)}</div>
              <p>Monthly savings</p>
            </div>
          </div>

          <div className="cost-visualization">
            <div className="chart-section">
              <h3>Cost Breakdown by Token Type</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={entry => `${entry.name}: $${entry.value.toFixed(2)}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="breakdown-details">
              <h3>Detailed Breakdown</h3>
              <table>
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Cost per Request</th>
                    <th>Monthly Cost</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Cached Input Tokens</td>
                    <td>${costData.breakdown.cached_input_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.cached_input_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Uncached Input Tokens</td>
                    <td>${costData.breakdown.uncached_input_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.uncached_input_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Output Tokens</td>
                    <td>${costData.breakdown.output_cost_per_request.toFixed(8)}</td>
                    <td>${(costData.breakdown.output_cost_per_request * requestsPerDay * 30).toFixed(2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CostCalculator;
