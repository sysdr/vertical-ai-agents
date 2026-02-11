import React, { useState } from 'react';
import axios from 'axios';
import './ValidatorDashboard.css';

const ValidatorDashboard = ({ apiUrl }) => {
  const [documents, setDocuments] = useState([
    { id: 'doc1', content: 'Our Q3 revenue was $50 million, up 20% from Q2.', score: 0.95, source: 'Financial Report', metadata: {} },
    { id: 'doc2', content: 'Third quarter earnings reached $50M, showing strong growth.', score: 0.92, source: 'Press Release', metadata: {} },
    { id: 'doc3', content: 'Q3 revenue totaled $48 million according to preliminary figures.', score: 0.88, source: 'Internal Memo', metadata: {} }
  ]);
  
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleValidate = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${apiUrl}/validator/validate`, documents);
      setReport(response.data);
    } catch (err) {
      const isConnectionRefused = !err.response && (err.code === 'ERR_NETWORK' || err.message?.includes('Network Error'));
      const message = isConnectionRefused
        ? 'Backend not running (connection refused). Start it from project root: ./build.sh then ./start.sh'
        : (err.response?.data?.detail || err.message);
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const addDocument = () => {
    const newDoc = {
      id: `doc${documents.length + 1}`,
      content: '',
      score: 0.9,
      source: 'User Input',
      metadata: {}
    };
    setDocuments([...documents, newDoc]);
  };

  const updateDocument = (index, field, value) => {
    const updated = [...documents];
    updated[index][field] = value;
    setDocuments(updated);
  };

  const removeDocument = (index) => {
    setDocuments(documents.filter((_, i) => i !== index));
  };

  const getStatusColor = (status) => {
    switch(status) {
      case 'passed': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'failed': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  return (
    <div className="validator-dashboard">
      {/* Input Section */}
      <div className="section">
        <div className="section-header">
          <h2>📄 Documents to Validate</h2>
          <button onClick={addDocument} className="btn-add">+ Add Document</button>
        </div>
        
        <div className="documents-input">
          {documents.map((doc, index) => (
            <div key={doc.id} className="document-card">
              <div className="document-header">
                <span className="document-id">{doc.id}</span>
                <button onClick={() => removeDocument(index)} className="btn-remove">✕</button>
              </div>
              <textarea
                value={doc.content}
                onChange={(e) => updateDocument(index, 'content', e.target.value)}
                placeholder="Enter document content..."
                rows={3}
              />
              <div className="document-meta">
                <input
                  type="text"
                  value={doc.source}
                  onChange={(e) => updateDocument(index, 'source', e.target.value)}
                  placeholder="Source"
                  className="input-source"
                />
                <input
                  type="number"
                  value={doc.score}
                  onChange={(e) => updateDocument(index, 'score', parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.01"
                  className="input-score"
                />
              </div>
            </div>
          ))}
        </div>
        
        <button 
          onClick={handleValidate} 
          disabled={loading || documents.length < 2}
          className="btn-validate"
        >
          {loading ? '🔄 Validating...' : '🔍 Validate Consistency'}
        </button>
        
        {error && (
          <div className="error-message">
            ⚠️ Error: {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {report && (
        <>
          <div className="section">
            <h2>📊 Validation Results</h2>
            
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Overall Consistency</div>
                <div className="metric-value" style={{color: report.overall_consistency > 0.7 ? '#4caf50' : report.overall_consistency > 0.5 ? '#ff9800' : '#f44336'}}>
                  {(report.overall_consistency * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Pairs Validated</div>
                <div className="metric-value">{report.validated_pairs}/{report.total_pairs}</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Cache Hit Rate</div>
                <div className="metric-value">{(report.cache_hit_rate * 100).toFixed(1)}%</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-label">Processing Time</div>
                <div className="metric-value">{(report.processing_time_ms / 1000).toFixed(2)}s</div>
              </div>
            </div>
          </div>

          <div className="section">
            <h2>📋 Validated Documents</h2>
            
            <div className="validated-documents">
              {report.documents.map((doc, index) => (
                <div key={doc.id} className="validated-doc-card">
                  <div className="validated-doc-header">
                    <span className="doc-id">{doc.id}</span>
                    <span 
                      className="doc-status"
                      style={{backgroundColor: getStatusColor(doc.validation_status)}}
                    >
                      {doc.validation_status}
                    </span>
                  </div>
                  
                  <div className="doc-content">{doc.content}</div>
                  
                  <div className="doc-metrics">
                    <span>Consistency: <strong>{(doc.consistency_score * 100).toFixed(1)}%</strong></span>
                    <span>Source: <strong>{doc.source}</strong></span>
                  </div>
                  
                  {doc.contradiction_flags.length > 0 && (
                    <div className="contradictions">
                      <strong>⚠️ Contradictions Found:</strong>
                      <ul>
                        {doc.contradiction_flags.map((flag, i) => (
                          <li key={i}>{flag}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {report.high_conflict_pairs.length > 0 && (
            <div className="section">
              <h2>🚨 High-Conflict Pairs</h2>
              
              <div className="conflict-pairs">
                {report.high_conflict_pairs.map((pair, index) => (
                  <div key={index} className="conflict-card">
                    <div className="conflict-header">
                      <span>{pair.doc1_id} ↔ {pair.doc2_id}</span>
                      <span className="conflict-score">
                        {(pair.consistency_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    {pair.contradictions.length > 0 && (
                      <ul className="conflict-list">
                        {pair.contradictions.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ValidatorDashboard;
