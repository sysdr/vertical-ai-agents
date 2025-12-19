import React, { useState, useEffect } from 'react';

function DecisionTraceViewer() {
  const [traces, setTraces] = useState([]);

  useEffect(() => {
    // In production, fetch real traces from backend
    setTraces([
      {
        id: 1,
        timestamp: new Date().toISOString(),
        state: 'planning',
        action: 'generate_plan',
        reasoning: 'Generated execution plan with 3 steps',
        duration_ms: 234.5
      },
      {
        id: 2,
        timestamp: new Date().toISOString(),
        state: 'executing',
        action: 'execute_calculator',
        reasoning: 'Calculating factorial of 5',
        duration_ms: 45.2
      }
    ]);
  }, []);

  return (
    <div className="trace-viewer">
      <h2>Decision Trace Viewer</h2>
      <p className="info-text">
        Traces show the agent's reasoning and decision-making process
      </p>

      <div className="trace-timeline">
        {traces.map((trace) => (
          <div key={trace.id} className="timeline-item">
            <div className="timeline-marker" data-state={trace.state}></div>
            <div className="timeline-content">
              <div className="timeline-header">
                <span className="timeline-state">{trace.state}</span>
                <span className="timeline-time">
                  {new Date(trace.timestamp).toLocaleTimeString()}
                </span>
                <span className="timeline-duration">{trace.duration_ms.toFixed(2)}ms</span>
              </div>
              <div className="timeline-action">{trace.action}</div>
              <div className="timeline-reasoning">{trace.reasoning}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DecisionTraceViewer;
