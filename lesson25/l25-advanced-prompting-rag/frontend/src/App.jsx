import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [domain, setDomain] = useState('strict');
  const [testMode, setTestMode] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Stats fetch error:', error);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          domain,
          test_mode: testMode
        })
      });

      const data = await response.json();
      if (!response.ok) {
        setResult({ error: data.detail || data.message || ('Error ' + response.status) });
        return;
      }
      setResult(data);
    } catch (error) {
      console.error('Query error:', error);
      setResult({ error: error.message });
    } finally {
      setLoading(false);
      fetchStats(); // refresh dashboard metrics after demo execution
    }
  };

  const getStateColor = (state) => {
    switch (state) {
      case 'grounded': return '#10b981';
      case 'partial': return '#f59e0b';
      case 'rejected': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStateEmoji = (state) => {
    switch (state) {
      case 'grounded': return '✓';
      case 'partial': return '⚠';
      case 'rejected': return '✗';
      default: return '?';
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🎯 L25: Advanced Prompting for RAG</h1>
        <p>System Prompts That Prevent Hallucination</p>
      </header>

      {stats && (
        <div className="stats-bar">
          <div className="stat">
            <span className="stat-label">Templates:</span>
            <span className="stat-value">{stats.templates_available?.length || 0}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Documents:</span>
            <span className="stat-value">{stats.total_documents || 0}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Vector Store:</span>
            <span className="stat-value">{stats.vector_store_initialized ? '✓' : '✗'}</span>
          </div>
        </div>
      )}

      <div className="container">
        <div className="query-panel">
          <h2>Query Configuration</h2>
          
          <div className="form-group">
            <label>Question</label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about our knowledge base..."
              rows={4}
            />
          </div>

          <div className="form-group">
            <label>Grounding Domain</label>
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              <option value="strict">Strict (Legal/Medical)</option>
              <option value="moderate">Moderate (Support)</option>
              <option value="creative">Creative (Briefs)</option>
            </select>
            <small className="hint">
              {domain === 'strict' && '100% grounding required, citations mandatory'}
              {domain === 'moderate' && 'Prioritize context, allow general knowledge'}
              {domain === 'creative' && 'Use context as inspiration, expand freely'}
            </small>
          </div>

          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={testMode}
                onChange={(e) => setTestMode(e.target.checked)}
              />
              <span>Adversarial Test Mode (use misleading context)</span>
            </label>
          </div>

          <button
            onClick={handleQuery}
            disabled={loading || !question.trim()}
            className="query-button"
          >
            {loading ? 'Processing...' : 'Execute Query'}
          </button>
        </div>

        {result && (
          <div className="result-panel">
            <div className="result-header">
              <h2>Response Analysis</h2>
              {result.error ? (
                <div className="grounding-badge" style={{ backgroundColor: '#6b7280' }}>
                  ⚠ Service
                </div>
              ) : (
                <div
                  className="grounding-badge"
                  style={{ backgroundColor: getStateColor(result.grounding_state) }}
                >
                  {getStateEmoji(result.grounding_state)} {result.grounding_state?.toUpperCase()}
                </div>
              )}
            </div>

            {result.error ? (
              <div className="answer-section" style={{ borderLeft: '4px solid #f59e0b' }}>
                <h3>{result.error.includes('GEMINI_API_KEY') || result.error.includes('503') || result.error.includes('Service Unavailable') ? 'API key required' : 'Something went wrong'}</h3>
                <div className="answer-text" style={{ marginBottom: '0.75rem' }}>{result.error}</div>
                {(result.error.includes('GEMINI_API_KEY') || result.error.includes('503') || result.error.includes('Service Unavailable')) && (
                  <div className="notice-setup">
                    <ol>
                      <li>
                        <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer" className="notice-cta">Get an API key from Google AI Studio</a>
                      </li>
                      <li>In the project folder: <code>cp .env.example .env</code>, then set <code>GEMINI_API_KEY=your_key</code> in <code>.env</code>.</li>
                      <li>Restart the app: <code>./stop.sh</code> then <code>./start.sh</code>.</li>
                    </ol>
                  </div>
                )}
              </div>
            ) : (
              <>
            <div className="answer-section">
              <h3>Generated Answer</h3>
              <div className="answer-text">{result.answer}</div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Grounding Confidence</div>
                <div className="metric-value">
                  {(result.grounding_confidence * 100).toFixed(1)}%
                </div>
                <div className="metric-bar">
                  <div
                    className="metric-fill"
                    style={{
                      width: `${result.grounding_confidence * 100}%`,
                      backgroundColor: getStateColor(result.grounding_state)
                    }}
                  />
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Citations Found</div>
                <div className="metric-value">{result.citations?.length || 0}</div>
              </div>

              <div className="metric-card">
                <div className="metric-label">Template Used</div>
                <div className="metric-value">{result.metadata?.domain || 'N/A'}</div>
              </div>
            </div>

            {result.citations && result.citations.length > 0 && (
              <div className="citations-section">
                <h3>Source Citations</h3>
                {result.citations.map((citation, idx) => (
                  <div key={idx} className="citation-card">
                    <div className="citation-header">
                      <span className="citation-id">Source {citation.source_id}</span>
                      <span className="citation-score">
                        Score: {citation.relevance_score.toFixed(3)}
                      </span>
                    </div>
                    <div className="citation-text">{citation.text}</div>
                  </div>
                ))}
              </div>
            )}

            {result.validation_results && (
              <div className="validation-section">
                <h3>Validation Results</h3>
                <pre className="validation-json">
                  {JSON.stringify(result.validation_results, null, 2)}
                </pre>
              </div>
            )}
              </>
            )}
          </div>
        )}
      </div>

      <footer className="footer">
        <p>L25: Advanced Prompting for RAG • VAIA Curriculum</p>
        <p>Building on L24's modular RAG • Enabling L26's multi-turn conversations</p>
      </footer>
    </div>
  );
}

export default App;
