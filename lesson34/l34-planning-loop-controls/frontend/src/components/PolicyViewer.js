import React from 'react';

function PolicyViewer({ policies }) {
  return (
    <div className="card policy-card">
      <h2>📋 Budget Policies</h2>
      
      {Object.keys(policies).length === 0 ? (
        <p className="loading">Loading policies...</p>
      ) : (
        <div className="policies-grid">
          {Object.entries(policies).map(([env, policy]) => (
            <div key={env} className="policy-item">
              <h3>{env.charAt(0).toUpperCase() + env.slice(1)}</h3>
              <div className="policy-details">
                <div className="detail">
                  <span className="icon">🔄</span>
                  <span className="label">Max Iterations:</span>
                  <span className="value">{policy.max_iterations}</span>
                </div>
                <div className="detail">
                  <span className="icon">🎫</span>
                  <span className="label">Max Tokens:</span>
                  <span className="value">{policy.max_tokens_per_turn.toLocaleString()}</span>
                </div>
                <div className="detail">
                  <span className="icon">⚠️</span>
                  <span className="label">Mode:</span>
                  <span className="value">{policy.enforcement_mode}</span>
                </div>
              </div>
              <p className="description">{policy.description}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default PolicyViewer;
