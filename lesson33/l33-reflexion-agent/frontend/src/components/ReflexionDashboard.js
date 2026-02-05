import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ReflexionDashboard.css';

const API_BASE = 'http://localhost:8000/api';

const API_KEY_ERROR_PHRASES = ['api key', 'api_key', 'renew the api key', 'invalid', 'expired'];
const MODEL_ERROR_PHRASES = ['model', 'not found', '404', 'not supported for generatecontent'];
const QUOTA_ERROR_PHRASES = ['429', 'quota', 'rate limit', 'exceeded your current quota'];

const ReflexionDashboard = () => {
  const [task, setTask] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [tools, setTools] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [apiKeyOk, setApiKeyOk] = useState(null);

  useEffect(() => {
    fetchTools();
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const r = await axios.get(`${API_BASE}/health`);
      setApiKeyOk(r.data.api_key_configured !== false);
    } catch {
      setApiKeyOk(null);
    }
  };

  const fetchTools = async () => {
    try {
      const response = await axios.get(`${API_BASE}/tools`);
      setTools(response.data.tools);
    } catch (error) {
      console.error('Failed to fetch tools:', error);
    }
  };

  const executeTask = async () => {
    if (!task.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/execute`, {
        task: task,
        session_id: sessionId
      });

      setResult(response.data);
      setSessionId(response.data.session_id);
    } catch (error) {
      setResult({
        success: false,
        result: error.response?.data?.detail || 'Request failed',
        attempts: 0,
        reflections: [],
        stats: {}
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeTask();
    }
  };

  const exampleTasks = [
    "Find the CEO of Anthropic",
    "What is the stock price of Google?",
    "Calculate 15 * 23 + 47",
    "Tell me about Reflexion AI technique"
  ];

  return (
    <div className="dashboard-container">
      <div className="dashboard-card">
        <header className="dashboard-header">
          <h1>🧠 L33: Reflexion Agent</h1>
          <p className="subtitle">Self-Correcting AI with Iterative Reflection</p>
          {apiKeyOk === false && (
            <div className="api-key-alert">
              <strong>API key required.</strong> Add your Gemini key to <code>l33-reflexion-agent/.env</code>:
              <code className="env-line">GEMINI_API_KEY=your_key</code>
                <span className="alert-actions">
                Get key: <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a>
                {' · '}
                Add to <code>.env</code>: GEMINI_API_KEY=your_key
              </span>
            </div>
          )}
        </header>

        <div className="tools-section">
          <h3>Available Tools</h3>
          <div className="tools-grid">
            {tools.map((tool, idx) => (
              <div key={idx} className="tool-card">
                <span className="tool-name">{tool.name}</span>
                <span className="tool-desc">{tool.description}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="input-section">
          <h3>Task Input</h3>
          <textarea
            className="task-input"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your task... (e.g., 'Find the CEO of Anthropic')"
            rows="3"
          />
          
          <div className="example-tasks">
            <span className="example-label">Try:</span>
            {exampleTasks.map((example, idx) => (
              <button
                key={idx}
                className="example-btn"
                onClick={() => setTask(example)}
              >
                {example}
              </button>
            ))}
          </div>

          <button
            className="execute-btn"
            onClick={executeTask}
            disabled={loading || !task.trim()}
          >
            {loading ? '🔄 Processing...' : '▶️ Execute with Reflexion'}
          </button>
        </div>

        {result && (
          <div className={`result-section ${result.success ? 'success' : 'failure'}`}>
            <div className="result-header">
              <h3>
                {result.success ? '✅ Success' : '❌ Failed'}
                <span className="attempts-badge">
                  {result.attempts} attempt{result.attempts !== 1 ? 's' : ''}
                </span>
              </h3>
            </div>

            <div className="result-content">
              <h4>Result:</h4>
              {result.error === 'API_KEY_INVALID' || (result.result && API_KEY_ERROR_PHRASES.some(p => result.result.toLowerCase().includes(p))) ? (
                <div className="api-key-error-box">
                  <strong>🔑 API Key Issue</strong>
                  <p>{result.result}</p>
                  <p className="setup-hint">Fix: Add a valid GEMINI_API_KEY to <code>l33-reflexion-agent/.env</code></p>
                  <p className="setup-hint">Get a key: <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a></p>
                </div>
              ) : result.error === 'QUOTA_EXCEEDED' || (result.result && QUOTA_ERROR_PHRASES.some(p => result.result.toLowerCase().includes(p))) ? (
                <div className="api-key-error-box">
                  <strong>⏱️ Rate Limit</strong>
                  <p>{result.result}</p>
                  <p className="setup-hint">Free tier: ~5 requests/min. Wait 1 min or upgrade at <a href="https://ai.google.dev/pricing" target="_blank" rel="noopener noreferrer">Google AI pricing</a></p>
                </div>
              ) : result.error === 'MODEL_NOT_FOUND' || (result.result && MODEL_ERROR_PHRASES.some(p => result.result.toLowerCase().includes(p)) && result.result.toLowerCase().includes('model')) ? (
                <div className="api-key-error-box">
                  <strong>📦 Model Not Found</strong>
                  <p>{result.result}</p>
                  <p className="setup-hint">Fix: Update .env with a supported model: <code>GEMINI_MODEL_MAIN=gemini-2.5-flash</code></p>
                  <p className="setup-hint">See: <a href="https://ai.google.dev/gemini-api/docs/models" target="_blank" rel="noopener noreferrer">Available models</a></p>
                </div>
              ) : (
                <pre>{result.result}</pre>
              )}
            </div>

            {result && (
              <div className="stats-section">
                <h4>Statistics:</h4>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Attempts:</span>
                    <span className="stat-value">{result.attempts ?? 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Reflections:</span>
                    <span className="stat-value">{result.stats?.total ?? 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Success Rate:</span>
                    <span className="stat-value">
                      {result.stats?.success_rate != null
                        ? ((result.stats.success_rate) * 100).toFixed(0) + '%'
                        : result.success ? '100%' : '0%'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg Confidence:</span>
                    <span className="stat-value">
                      {result.stats?.avg_confidence != null
                        ? ((result.stats.avg_confidence) * 100).toFixed(0) + '%'
                        : result.success ? '100%' : '-'}
                    </span>
                  </div>
                  {result.stats?.failed != null && result.stats.failed > 0 && (
                    <div className="stat-item">
                      <span className="stat-label">Failed Reflections:</span>
                      <span className="stat-value">{result.stats.failed}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {result.reflections && result.reflections.length > 0 && (
              <div className="reflections-section">
                <h4>Reflection History:</h4>
                {result.reflections.map((ref, idx) => (
                  <div key={idx} className="reflection-card">
                    <div className="reflection-header">
                      <span className="reflection-attempt">Attempt {ref.attempt}</span>
                      <span className={`reflection-status ${ref.success ? 'success' : 'failed'}`}>
                        {ref.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                    <div className="reflection-body">
                      <div className="reflection-item">
                        <strong>Action:</strong>
                        <code>{ref.action}</code>
                      </div>
                      <div className="reflection-item">
                        <strong>Critique:</strong>
                        <p>{ref.critique}</p>
                      </div>
                      <div className="reflection-item">
                        <strong>Next Strategy:</strong>
                        <p>{ref.next_strategy}</p>
                      </div>
                      {ref.confidence !== undefined && (
                        <div className="reflection-item">
                          <strong>Confidence:</strong>
                          <div className="confidence-bar">
                            <div 
                              className="confidence-fill" 
                              style={{width: `${ref.confidence * 100}%`}}
                            />
                            <span className="confidence-text">
                              {(ref.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <footer className="dashboard-footer">
          <p>Lesson 33: Implementing Self-Correction (Reflexion)</p>
          <p className="session-id">Session ID: {sessionId || 'Not started'}</p>
        </footer>
      </div>
    </div>
  );
};

export default ReflexionDashboard;
