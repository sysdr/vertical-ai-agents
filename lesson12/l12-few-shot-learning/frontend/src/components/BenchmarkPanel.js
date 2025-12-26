import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function BenchmarkPanel() {
  const [queries, setQueries] = useState('Customer wants a refund\nWhen will my order arrive?\nI want to speak to a manager\nThank you for great service\nProduct came damaged');
  const [taskDescription, setTaskDescription] = useState('Classify customer support messages');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRunBenchmark = async () => {
    const queryList = queries.split('\n').filter(q => q.trim());
    if (queryList.length === 0) {
      alert('Please enter at least one query');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/benchmark', {
        queries: queryList,
        task_description: taskDescription,
        domain: 'customer_support'
      });
      setResults(response.data);
    } catch (error) {
      console.error('Benchmark failed:', error);
      alert('Benchmark failed. See console for details.');
    } finally {
      setLoading(false);
    }
  };

  const chartData = results ? results.map(r => ({
    shots: `${r.shot_count}-shot`,
    accuracy: (r.accuracy * 100).toFixed(1),
    confidence: (r.avg_confidence * 100).toFixed(1),
    latency: r.avg_latency_ms.toFixed(0)
  })) : [];

  return (
    <div className="panel benchmark-panel">
      <h2>Performance Benchmark</h2>
      <p className="info">Test classification accuracy across different shot counts (0, 1, 3, 5)</p>

      <div className="form-group">
        <label>Test Queries (one per line)</label>
        <textarea
          value={queries}
          onChange={(e) => setQueries(e.target.value)}
          rows="6"
          placeholder="Enter test queries, one per line..."
        />
      </div>

      <div className="form-group">
        <label>Task Description</label>
        <input
          type="text"
          value={taskDescription}
          onChange={(e) => setTaskDescription(e.target.value)}
          placeholder="Describe the classification task..."
        />
      </div>

      <button 
        onClick={handleRunBenchmark}
        disabled={loading}
        className="btn-primary"
      >
        {loading ? '🔄 Running Benchmark...' : '▶️ Run Benchmark'}
      </button>

      {results && (
        <div className="benchmark-results">
          <h3>Benchmark Results</h3>
          
          <div className="results-grid">
            {results.map((result) => (
              <div key={result.shot_count} className="result-card">
                <h4>{result.shot_count}-Shot</h4>
                <div className="metric">
                  <span className="label">Accuracy:</span>
                  <span className="value">{(result.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <span className="label">Confidence:</span>
                  <span className="value">{(result.avg_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="metric">
                  <span className="label">Avg Latency:</span>
                  <span className="value">{result.avg_latency_ms.toFixed(0)}ms</span>
                </div>
                <div className="metric">
                  <span className="label">Avg Tokens:</span>
                  <span className="value">{result.avg_tokens}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="chart-container">
            <h4>Accuracy Comparison</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="shots" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#4CAF50" strokeWidth={2} name="Accuracy %" />
                <Line type="monotone" dataKey="confidence" stroke="#2196F3" strokeWidth={2} name="Confidence %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default BenchmarkPanel;
