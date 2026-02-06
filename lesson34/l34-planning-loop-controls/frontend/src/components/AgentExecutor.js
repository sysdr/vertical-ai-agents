import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

function AgentExecutor({ onExecutionComplete }) {
  const [task, setTask] = useState('');
  const [environment, setEnvironment] = useState('production');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const executeTask = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/agent/execute`, {
        task,
        environment
      });
      setResult(response.data);
      onExecutionComplete();
    } catch (error) {
      setResult({
        success: false,
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card executor-card">
      <h2>🤖 Execute Agent Task</h2>
      
      <form onSubmit={executeTask}>
        <div className="form-group">
          <label>Task Description:</label>
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Enter a task for the agent to complete..."
            rows="4"
            required
          />
        </div>

        <div className="form-group">
          <label>Environment:</label>
          <select 
            value={environment} 
            onChange={(e) => setEnvironment(e.target.value)}
          >
            <option value="development">Development (50 iter, 500K tokens)</option>
            <option value="staging">Staging (20 iter, 100K tokens)</option>
            <option value="production">Production (10 iter, 50K tokens)</option>
          </select>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? '⏳ Executing...' : '▶️ Execute Task'}
        </button>
      </form>

      {result && (
        <div className={`result ${result.success ? 'success' : 'error'}`}>
          <h3>{result.success ? '✅ Success' : '❌ Failed'}</h3>
          
          {result.result && (
            <div className="result-text">
              <strong>Result:</strong>
              <p>{result.result}</p>
            </div>
          )}
          
          {result.error && (
            <div className="error-text">
              <strong>Error:</strong>
              <p>{result.error}</p>
            </div>
          )}
          
          {result.metrics && (
            <div className="metrics-summary">
              <h4>📊 Execution Metrics:</h4>
              <div className="metrics-grid">
                <div className="metric">
                  <span className="label">Iterations:</span>
                  <span className="value">{result.metrics.total_iterations} / {result.metrics.max_iterations}</span>
                </div>
                <div className="metric">
                  <span className="label">Tokens:</span>
                  <span className="value">{result.metrics.total_tokens.toLocaleString()} / {result.metrics.max_tokens.toLocaleString()}</span>
                </div>
                <div className="metric">
                  <span className="label">Steps:</span>
                  <span className="value">{result.metrics.steps_completed}</span>
                </div>
                <div className="metric">
                  <span className="label">Duration:</span>
                  <span className="value">{result.metrics.elapsed_seconds.toFixed(2)}s</span>
                </div>
              </div>
              
              <div className="utilization-bars">
                <div className="bar-group">
                  <label>Iteration Utilization: {(result.metrics.iteration_utilization * 100).toFixed(1)}%</label>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${result.metrics.iteration_utilization * 100}%` }}
                    />
                  </div>
                </div>
                
                <div className="bar-group">
                  <label>Token Utilization: {(result.metrics.token_utilization * 100).toFixed(1)}%</label>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${result.metrics.token_utilization * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AgentExecutor;
