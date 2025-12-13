import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function ContextAnalyzer({ models }) {
  const [selectedModel, setSelectedModel] = useState('gemini-2.0-flash-exp');
  const [sampleTexts, setSampleTexts] = useState('');
  const [contextData, setContextData] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeContext = async () => {
    const texts = sampleTexts.split('\n').filter(t => t.trim().length > 0);
    
    if (texts.length === 0) {
      alert('Please enter at least one sample text (one per line)');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/analyze/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: selectedModel,
          sample_texts: texts
        })
      });
      const data = await response.json();
      setContextData(data);
    } catch (error) {
      console.error('Error analyzing context:', error);
      alert('Context analysis failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const distributionData = contextData ? [
    { name: 'Min', value: contextData.token_distribution.min },
    { name: 'Mean', value: contextData.token_distribution.mean },
    { name: 'Median', value: contextData.token_distribution.median },
    { name: 'P95', value: contextData.token_distribution.p95 },
    { name: 'P99', value: contextData.token_distribution.p99 },
    { name: 'Max', value: contextData.token_distribution.max }
  ] : [];

  return (
    <div className="context-analyzer">
      <div className="section-header">
        <h2>Context Window Analyzer</h2>
        <p>Optimize context usage and identify cost savings opportunities</p>
      </div>

      <div className="analyzer-form">
        <div className="form-group">
          <label>Select Model</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {Object.keys(models).map(modelName => (
              <option key={modelName} value={modelName}>{modelName}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Sample Texts (one per line)</label>
          <textarea 
            value={sampleTexts}
            onChange={(e) => setSampleTexts(e.target.value)}
            rows={10}
            placeholder="Enter sample texts to analyze token distribution...
Example:
What is machine learning?
Explain the benefits of cloud computing
How do I optimize my database queries?"
          />
        </div>

        <button 
          className="analyze-btn" 
          onClick={analyzeContext} 
          disabled={loading}
        >
          {loading ? 'Analyzing...' : '📝 Analyze Context Usage'}
        </button>
      </div>

      {contextData && (
        <div className="context-results">
          <div className="context-summary">
            <h3>Analysis Summary</h3>
            <div className="summary-grid">
              <div className="summary-item">
                <span className="label">Samples Analyzed:</span>
                <span className="value">{contextData.samples_analyzed}</span>
              </div>
              <div className="summary-item">
                <span className="label">Model Context Limit:</span>
                <span className="value">{(contextData.context_window_max / 1000).toFixed(0)}K tokens</span>
              </div>
              <div className="summary-item">
                <span className="label">Recommended Limit:</span>
                <span className="value">{contextData.recommended_context_limit} tokens</span>
              </div>
              <div className="summary-item highlight">
                <span className="label">Potential Savings:</span>
                <span className="value">{contextData.efficiency_metrics.potential_savings_pct.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="distribution-chart">
            <h3>Token Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} label={{ value: 'Tokens', angle: -90, position: 'insideLeft' }} />
                <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd' }} />
                <Bar dataKey="value" fill="#4CAF50" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="efficiency-metrics">
            <h3>Efficiency Metrics</h3>
            <table>
              <tbody>
                <tr>
                  <td>Context Window Utilization</td>
                  <td><strong>{contextData.efficiency_metrics.utilization_rate.toFixed(2)}%</strong></td>
                </tr>
                <tr>
                  <td>Potential Savings Percentage</td>
                  <td><strong>{contextData.efficiency_metrics.potential_savings_pct.toFixed(2)}%</strong></td>
                </tr>
                <tr>
                  <td>Tokens Saved per Request</td>
                  <td><strong>{contextData.efficiency_metrics.tokens_saved_per_request.toLocaleString()}</strong></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="recommendations">
            <h3>💡 Recommendations</h3>
            <ul>
              {contextData.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default ContextAnalyzer;
