import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [evaluationRunning, setEvaluationRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [regressions, setRegressions] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
    fetchRegressions();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/metrics/summary');
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError('Cannot connect to backend. Start the backend server.');
    }
  };

  const fetchRegressions = async () => {
    try {
      const response = await fetch('/api/regression/detect', { method: 'POST' });
      const data = await response.json();
      setRegressions(Array.isArray(data) ? data : []);
    } catch (err) {
      setRegressions([]);
    }
  };

  const runEvaluation = async () => {
    setEvaluationRunning(true);
    setError(null);
    try {
      const response = await fetch('/api/evaluate/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_suite_name: 'default', iterations: 3 })
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setResults(data);
      await fetchMetrics();
      await fetchRegressions();
    } catch (err) {
      setError(err.message || 'Evaluation failed');
    }
    setEvaluationRunning(false);
  };

  const latencyData = results?.latency_summary?.map(l => ({
    component: l.component,
    P50: l.p50_ms,
    P95: l.p95_ms,
    P99: l.p99_ms
  })) || [];

  const hasMetrics = metrics && (metrics.total_tests_run > 0 || (metrics.tool_call_accuracy + metrics.rag_groundedness + metrics.p99_latency_ms) > 0);

  return (
    <div className="app">
      <header className="header">
        <h1>🧪 L29: Evaluation Dashboard</h1>
        <p>Tool Reliability • RAG Groundedness • System Latency</p>
      </header>

      <div className="container">
        {error && <div className="error-banner">{error}</div>}

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{metrics ? (hasMetrics ? (metrics.overall_accuracy * 100).toFixed(1) : '—') : '...'}%</div>
            <div className="metric-label">Overall Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics ? (hasMetrics ? (metrics.tool_call_accuracy * 100).toFixed(1) : '—') : '...'}%</div>
            <div className="metric-label">Tool Call Accuracy</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics ? (hasMetrics ? (metrics.rag_groundedness * 100).toFixed(1) : '—') : '...'}%</div>
            <div className="metric-label">RAG Groundedness</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics ? (hasMetrics ? Math.round(metrics.p99_latency_ms) + 'ms' : '—') : '...'}</div>
            <div className="metric-label">P99 Latency</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{metrics ? (metrics.total_tests_run ?? '—') : '...'}</div>
            <div className="metric-label">Tests Run</div>
          </div>
        </div>

        {!hasMetrics && metrics && (
          <div className="panel prompt-panel">
            <p>Run an evaluation to populate metrics and see results on the dashboard.</p>
          </div>
        )}

        <div className="panel">
          <h2>Evaluation Controls</h2>
          <button className="btn-primary" onClick={runEvaluation} disabled={evaluationRunning}>
            {evaluationRunning ? '⏳ Running Evaluation...' : '▶️ Run Full Evaluation'}
          </button>
        </div>

        {results && (
          <div className="panel">
            <h2>Latest Evaluation Results</h2>
            <div className="results-grid">
              <div className="result-item">
                <span className="result-label">Test Suite:</span>
                <span className="result-value">{results.test_suite}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Total Tests:</span>
                <span className="result-value">{results.total_tests}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Passed:</span>
                <span className="result-value success">{results.passed}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Failed:</span>
                <span className="result-value error">{results.failed}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Duration:</span>
                <span className="result-value">{results.duration_seconds?.toFixed(2)}s</span>
              </div>
            </div>
            {latencyData.length > 0 && (
              <div className="chart-container">
                <h3>Latency Percentiles by Component</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={latencyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="component" stroke="#666" />
                    <YAxis stroke="#666" />
                    <Tooltip contentStyle={{ background: '#fff', border: '1px solid #ddd' }} />
                    <Legend />
                    <Bar dataKey="P50" fill="#4ade80" />
                    <Bar dataKey="P95" fill="#fbbf24" />
                    <Bar dataKey="P99" fill="#f87171" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {regressions.length > 0 && (
          <div className="panel">
            <h2>Regression Analysis</h2>
            <div className="regressions-list">
              {regressions.map((reg, idx) => (
                <div key={idx} className={`regression-item ${reg.is_regression ? 'regression' : 'ok'}`}>
                  <div className="regression-metric">{reg.metric_name}</div>
                  <div className="regression-values">
                    <span>Current: {reg.current_value.toFixed(3)}</span>
                    <span>Baseline: {reg.baseline_value.toFixed(3)}</span>
                    <span className={reg.change_percent < 0 ? 'negative' : 'positive'}>
                      {reg.change_percent > 0 ? '+' : ''}{reg.change_percent.toFixed(2)}%
                    </span>
                  </div>
                  {reg.is_regression && <div className="regression-alert">⚠️ Regression detected</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
