import React, { useState } from 'react';
import './ValidationForm.css';

// Show user-friendly issue text instead of raw regex
function formatIssue(issue) {
  if (typeof issue !== 'string') return String(issue);
  if (issue.startsWith('Compliance: Pattern ')) return 'Compliance: PII detected (e.g. SSN, credit card, passport number)';
  if (issue.startsWith('Factual check error:')) return issue; // keep API/model errors as-is
  return issue;
}

const SAMPLE_DOCUMENTS = [
  {
    id: "doc1",
    content: "Patient John Doe was prescribed ibuprofen 400mg for pain management. Follow up in 2 weeks.",
    source: "Medical Records"
  },
  {
    id: "doc2",
    content: "Treatment shows 95% effectiveness based on peer-reviewed study (JAMA 2024). Patients should consult their healthcare provider before starting any new medication.",
    source: "Clinical Study"
  }
];

function ValidationForm({ onValidate, loading, result }) {
  const [documents, setDocuments] = useState(JSON.stringify(SAMPLE_DOCUMENTS, null, 2));
  const [query, setQuery] = useState("What are the treatment options for chronic pain?");
  const [domain, setDomain] = useState("healthcare");

  const handleSubmit = (e) => {
    e.preventDefault();
    
    try {
      const parsedDocs = JSON.parse(documents);
      onValidate(parsedDocs, query, domain);
    } catch (error) {
      alert('Invalid JSON format for documents');
    }
  };

  const loadSample = () => {
    setDocuments(JSON.stringify(SAMPLE_DOCUMENTS, null, 2));
    setQuery("What are the treatment options for chronic pain?");
  };

  return (
    <div className="validation-form">
      <h2>🔍 Test Validation</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Domain</label>
          <select value={domain} onChange={(e) => setDomain(e.target.value)}>
            <option value="healthcare">Healthcare</option>
            <option value="financial">Financial</option>
            <option value="all">All Domains</option>
          </select>
        </div>

        <div className="form-group">
          <label>Query</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter search query..."
          />
        </div>

        <div className="form-group">
          <label>
            Documents (JSON)
            <button type="button" onClick={loadSample} className="load-sample-btn">
              Load Sample
            </button>
          </label>
          <textarea
            value={documents}
            onChange={(e) => setDocuments(e.target.value)}
            rows={8}
            placeholder='[{"id": "doc1", "content": "...", "source": "..."}]'
          />
        </div>

        <button type="submit" disabled={loading} className="validate-btn">
          {loading ? '⏳ Validating...' : '✅ Validate Content'}
        </button>
      </form>

      {result && (
        <div className={`validation-result ${result.approved ? 'approved' : 'rejected'}`}>
          <div className="result-header">
            <span className="result-icon">{result.approved ? '✅' : '❌'}</span>
            <h3>{result.approved ? 'Content Approved' : 'Content Rejected'}</h3>
          </div>
          
          {result.error ? (
            <div className="error-message">{result.error}</div>
          ) : (
            <>
              <div className="result-details">
                <div className="detail-item">
                  <strong>Factual Check:</strong>
                  <span className={result.factual_check_passed ? 'passed' : 'failed'}>
                    {result.factual_check_passed ? 'Passed' : 'Failed'}
                  </span>
                </div>
                <div className="detail-item">
                  <strong>Compliance Check:</strong>
                  <span className={result.compliance_check_passed ? 'passed' : 'failed'}>
                    {result.compliance_check_passed ? 'Passed' : 'Failed'}
                  </span>
                </div>
                <div className="detail-item">
                  <strong>Risk Score:</strong>
                  <span className={`risk-score risk-${result.risk_score > 70 ? 'high' : result.risk_score > 40 ? 'medium' : 'low'}`}>
                    {result.risk_score}
                  </span>
                </div>
                <div className="detail-item">
                  <strong>Audit ID:</strong>
                  <code>{result.audit_id}</code>
                </div>
              </div>

              {result.issues && result.issues.length > 0 && (
                <div className="issues-list">
                  <h4>Issues Detected:</h4>
                  <ul>
                    {result.issues.map((issue, idx) => (
                      <li key={idx}>{formatIssue(issue)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default ValidationForm;
