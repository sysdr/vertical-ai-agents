import React from 'react';

function MemoryViewer({ memory }) {
  if (memory.length === 0) {
    return <p className="empty-state">No reasoning traces yet. Submit a query to begin.</p>;
  }

  return (
    <div className="memory-viewer">
      {memory.slice().reverse().map((trace, i) => {
        const quality_scores = trace.quality_scores || {};
        const overall_quality = quality_scores.overall_quality || 0;
        const step_count = quality_scores.step_count || 0;
        const clarity_score = quality_scores.clarity_score || 0;
        const logic_flow = quality_scores.logic_flow || 0;
        
        return (
          <div key={i} className="memory-item">
            <div className="memory-header">
              <span className="memory-time">
                {trace.timestamp ? new Date(trace.timestamp).toLocaleString() : 'Unknown time'}
              </span>
              <span 
                className="quality-badge"
                style={{
                  background: overall_quality >= 0.7 
                    ? '#4caf50' 
                    : overall_quality >= 0.5 
                      ? '#ff9800' 
                      : '#f44336'
                }}
              >
                {(overall_quality * 100).toFixed(0)}%
              </span>
            </div>
            <p className="memory-query">{trace.query || 'No query'}</p>
            <div className="memory-stats">
              <span>Steps: {step_count}</span>
              <span>Clarity: {(clarity_score * 100).toFixed(0)}%</span>
              <span>Logic: {(logic_flow * 100).toFixed(0)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default MemoryViewer;
