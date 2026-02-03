import React from 'react';

function TraceViewer({ trace }) {
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4ade80';
    if (confidence >= 0.5) return '#fbbf24';
    return '#f87171';
  };

  return (
    <div className="trace-viewer">
      <div className="trace-header">
        <h2>Planning Trace</h2>
        <div className="trace-stats">
          <span className="stat">
            <strong>Status:</strong> {trace.status}
          </span>
          <span className="stat">
            <strong>Iterations:</strong> {trace.total_iterations}
          </span>
          <span className="stat">
            <strong>Total Tokens:</strong> {trace.total_tokens}
          </span>
          <span className="stat">
            <strong>Latency:</strong> {trace.total_latency_ms.toFixed(2)}ms
          </span>
        </div>
      </div>

      <div className="task-description">
        <strong>Task:</strong> {trace.task}
      </div>

      <div className="steps-container">
        {trace.steps.map((step, index) => (
          <div key={index} className="planning-step">
            <div className="step-header">
              <span className="step-number">Iteration {step.iteration}</span>
              <span className="step-time">{formatTimestamp(step.timestamp)}</span>
            </div>

            <div className="step-observation">
              <h4>Observation</h4>
              {step.observation.previous_action && (
                <p><strong>Previous Action:</strong> {step.observation.previous_action}</p>
              )}
              {Object.keys(step.observation.context).length > 0 && (
                <div className="context">
                  <strong>Context:</strong>
                  <pre>{JSON.stringify(step.observation.context, null, 2)}</pre>
                </div>
              )}
            </div>

            <div className="step-reasoning">
              <h4>
                Reasoning 
                <span 
                  className="confidence-badge"
                  style={{ backgroundColor: getConfidenceColor(step.reasoning.confidence) }}
                >
                  {(step.reasoning.confidence * 100).toFixed(0)}% confidence
                </span>
              </h4>
              <div className="thought">
                <strong>Thought:</strong> {step.reasoning.thought}
              </div>
              <div className="action">
                <strong>Action:</strong> {step.reasoning.action}
              </div>
              {step.reasoning.is_terminal && (
                <div className="terminal-flag">✓ Terminal Step</div>
              )}
            </div>

            <div className="step-metrics">
              <span>Latency: {step.metrics.latency_ms?.toFixed(2)}ms</span>
              <span>Tokens: ~{step.metrics.tokens_used}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TraceViewer;
