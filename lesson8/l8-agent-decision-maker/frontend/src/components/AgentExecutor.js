import React, { useState } from 'react';
import axios from 'axios';

function AgentExecutor() {
  const [goal, setGoal] = useState('Calculate 5 factorial');
  const [context, setContext] = useState('{}');
  const [maxSteps, setMaxSteps] = useState(10);
  const [timeout, setTimeout] = useState(30);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const presetGoals = [
    'Calculate 5 factorial',
    'Get weather for San Francisco',
    'Search for latest AI research',
    'Query database for user records'
  ];

  const executeGoal = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const contextObj = JSON.parse(context);
      const response = await axios.post('http://localhost:8000/agent/execute', {
        goal,
        context: contextObj,
        max_steps: maxSteps,
        timeout_seconds: timeout
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="executor-panel">
      <div className="input-section">
        <h2>Execute Agent Goal</h2>
        
        <div className="preset-goals">
          <label>Quick Presets:</label>
          <div className="preset-buttons">
            {presetGoals.map((preset, idx) => (
              <button
                key={idx}
                onClick={() => setGoal(preset)}
                className="preset-btn"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label>Goal:</label>
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            rows={3}
            placeholder="Enter agent goal..."
          />
        </div>

        <div className="form-group">
          <label>Context (JSON):</label>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            rows={4}
            placeholder='{"user_id": "123", "preferences": {}}'
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Max Steps:</label>
            <input
              type="number"
              value={maxSteps}
              onChange={(e) => setMaxSteps(parseInt(e.target.value))}
              min={1}
              max={50}
            />
          </div>
          <div className="form-group">
            <label>Timeout (seconds):</label>
            <input
              type="number"
              value={timeout}
              onChange={(e) => setTimeout(parseInt(e.target.value))}
              min={5}
              max={120}
            />
          </div>
        </div>

        <button
          onClick={executeGoal}
          disabled={loading || !goal}
          className="execute-btn"
        >
          {loading ? 'Executing...' : '🚀 Execute Goal'}
        </button>
      </div>

      {error && (
        <div className="error-box">
          <h3>❌ Error</h3>
          <pre>{error}</pre>
        </div>
      )}

      {result && (
        <div className="result-box">
          <h3>✅ Execution Complete</h3>
          <div className="result-summary">
            <div className="metric">
              <span>Status:</span>
              <span className={result.success ? 'success' : 'failed'}>
                {result.state}
              </span>
            </div>
            <div className="metric">
              <span>Duration:</span>
              <span>{result.total_duration_ms.toFixed(2)}ms</span>
            </div>
            <div className="metric">
              <span>Steps Completed:</span>
              <span>{result.results.steps_completed?.length || 0}</span>
            </div>
          </div>

          {result.plan && (
            <div className="plan-section">
              <h4>Generated Plan</h4>
              <div className="plan-info">
                <span>Steps: {result.plan.estimated_steps}</span>
                <span>Confidence: {(result.plan.confidence * 100).toFixed(0)}%</span>
              </div>
              <ol>
                {result.plan.steps.map((step, idx) => (
                  <li key={idx}>
                    <strong>{step.tool_name}</strong>: {step.description}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {result.traces && result.traces.length > 0 && (
            <div className="traces-section">
              <h4>Decision Traces ({result.traces.length})</h4>
              <div className="traces-list">
                {result.traces.map((trace, idx) => (
                  <div key={idx} className="trace-item">
                    <div className="trace-header">
                      <span className="trace-state">{trace.state}</span>
                      <span className="trace-action">{trace.action}</span>
                      <span className="trace-duration">{trace.duration_ms.toFixed(2)}ms</span>
                    </div>
                    <div className="trace-reasoning">{trace.reasoning}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <details className="raw-response">
            <summary>Raw Response</summary>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

export default AgentExecutor;
