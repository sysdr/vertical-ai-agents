import React from 'react';
import './AuditTrail.css';

// User-friendly violation messages (avoid showing raw regex in UI)
const RULE_LABELS = {
  PII_DETECTION: 'PII detected (e.g. SSN, credit card, passport number)',
  CITATION_REQUIRED: 'Missing required citation or reference',
  MISLEADING_CLAIM: 'Potentially misleading or unsubstantiated claim',
  FALSE_INFO: 'Demonstrably false or misleading information',
};

function formatViolationRationale(ruleId, rationale) {
  if (!rationale) return ruleId.replace(/_/g, ' ');
  if (/^Pattern\s+['`]/.test(rationale.trim())) {
    return RULE_LABELS[ruleId] || `${ruleId.replace(/_/g, ' ')}: pattern match in content`;
  }
  return rationale.length > 120 ? rationale.slice(0, 120) + '…' : rationale;
}

function AuditTrail({ trail }) {
  if (!trail || trail.length === 0) {
    return (
      <div className="audit-trail">
        <h2>📋 Audit Trail</h2>
        <p className="no-data">No audit entries yet. Run a validation to see results.</p>
      </div>
    );
  }

  return (
    <div className="audit-trail">
      <h2>📋 Audit Trail</h2>
      <p className="trail-info">Recent validation decisions (immutable compliance log)</p>
      
      <div className="trail-list">
        {trail.map((entry, idx) => (
          <div key={idx} className={`trail-entry ${entry.decision}`}>
            <div className="entry-header">
              <span className="entry-id">Audit ID: {entry.audit_id}</span>
              <span className="entry-timestamp">
                {new Date(entry.timestamp).toLocaleString()}
              </span>
            </div>
            
            <div className="entry-body">
              <div className="entry-info">
                <span className={`decision-badge ${entry.decision}`}>
                  {entry.decision.toUpperCase()}
                </span>
                <span className="risk-badge">
                  Risk: {entry.risk_score}
                </span>
                <span className="domain-badge">
                  {entry.domain}
                </span>
              </div>
              
              <div className="entry-content">
                <strong>Content:</strong> {entry.content_snippet}...
              </div>
              
              {entry.violations && entry.violations.length > 0 && (
                <div className="entry-violations">
                  <strong>Violations ({entry.violations.length}):</strong>
                  <ul>
                    {entry.violations.slice(0, 3).map((v, vidx) => (
                      <li key={vidx}>
                        <code>{v.rule_id}</code>: {formatViolationRationale(v.rule_id, v.rationale)}
                      </li>
                    ))}
                    {entry.violations.length > 3 && (
                      <li className="more">+{entry.violations.length - 3} more...</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AuditTrail;
