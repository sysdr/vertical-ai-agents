import React from 'react';

function ReasoningDisplay({ reasoning }) {
  // Add null checks and default values
  if (!reasoning) {
    return <div className="reasoning-display">No reasoning data available.</div>;
  }

  const { 
    query = '', 
    steps = [], 
    conclusion = '', 
    quality_scores = {} 
  } = reasoning;

  // Provide default values for quality_scores
  const {
    overall_quality = 0,
    step_count = 0,
    clarity_score = 0,
    logic_flow = 0,
    conclusion_present = false
  } = quality_scores;

  const getQualityColor = (score) => {
    if (score >= 0.7) return '#4caf50';
    if (score >= 0.5) return '#ff9800';
    return '#f44336';
  };

  const getQualityLabel = (score) => {
    if (score >= 0.7) return 'High Quality';
    if (score >= 0.5) return 'Medium Quality';
    return 'Low Quality';
  };

  return (
    <div className="reasoning-display">
      <div className="query-section">
        <h3>Query</h3>
        <p className="query-text">{query}</p>
      </div>

      <div className="quality-section">
        <h3>Quality Metrics</h3>
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-label">Overall Quality:</span>
            <div className="metric-value" style={{ color: getQualityColor(overall_quality) }}>
              {(overall_quality * 100).toFixed(0)}%
              <span className="quality-label">{getQualityLabel(overall_quality)}</span>
            </div>
          </div>
          <div className="metric">
            <span className="metric-label">Steps:</span>
            <span className="metric-value">{step_count}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Clarity:</span>
            <span className="metric-value">{(clarity_score * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Logic Flow:</span>
            <span className="metric-value">{(logic_flow * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Conclusion:</span>
            <span className="metric-value">{conclusion_present ? '✓' : '✗'}</span>
          </div>
        </div>
      </div>

      <div className="steps-section">
        <h3>Reasoning Steps ({Array.isArray(steps) ? steps.length : 0})</h3>
        {Array.isArray(steps) && steps.length > 0 ? (
          <ol className="steps-list">
            {steps.map((step, i) => (
              <li key={i} className="step-item">{step}</li>
            ))}
          </ol>
        ) : (
          <p className="empty-state">No steps available.</p>
        )}
      </div>

      {conclusion && (
        <div className="conclusion-section">
          <h3>Conclusion</h3>
          <p className="conclusion-text">{conclusion}</p>
        </div>
      )}
    </div>
  );
}

export default ReasoningDisplay;
