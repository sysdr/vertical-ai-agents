import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Loader, CheckCircle, XCircle, Brain, Wrench, Activity, Database, Zap } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 120000; // 2 min for slow queries

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [result, setResult] = useState(null);
  const elapsedTimerRef = useRef(null);
  const [tools, setTools] = useState([]);
  const [showTools, setShowTools] = useState(false);
  const [metrics, setMetrics] = useState({ tools_loaded: 0, active_sessions: 0, total_queries: 0, status: 'unknown' });

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      setMetrics({
        tools_loaded: response.data.tools_loaded ?? 0,
        active_sessions: response.data.active_sessions ?? 0,
        total_queries: response.data.total_queries ?? 0,
        status: response.data.status ?? 'unknown'
      });
    } catch (error) {
      setMetrics({ tools_loaded: 0, active_sessions: 0, total_queries: 0, status: 'offline' });
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchTools = async () => {
    try {
      const response = await axios.get(`${API_URL}/agent/tools`);
      setTools(response.data);
      setShowTools(!showTools);
    } catch (error) {
      console.error('Error fetching tools:', error);
    }
  };

  const executeQuery = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setResult(null);
    setElapsedSeconds(0);
    
    // Start elapsed time counter
    elapsedTimerRef.current = setInterval(() => {
      setElapsedSeconds(s => s + 1);
    }, 1000);
    
    try {
      const response = await axios.post(
        `${API_URL}/agent/query`,
        { query: query.trim(), max_iterations: 5 },
        { timeout: REQUEST_TIMEOUT_MS }
      );
      setResult(response.data);
      fetchMetrics();
    } catch (error) {
      console.error('Error executing query:', error);
      const detail = error.response?.data?.detail || error.message || 'Request failed';
      const detailStr = typeof detail === 'string' ? detail : JSON.stringify(detail);
      const isTimeout = error.code === 'ECONNABORTED' || detailStr.includes('timeout');
      const isApiKeyError = /API_KEY|api key expired|API key expired|invalid.*key/i.test(detailStr);
      setResult({
        success: false,
        error: isApiKeyError
          ? 'API key expired or invalid. Get a new key at https://aistudio.google.com/apikey and add GEMINI_API_KEY=your_key to .env, then restart.'
          : isTimeout
            ? 'Request timed out. Try a simpler query.'
            : detail
      });
    } finally {
      setLoading(false);
      if (elapsedTimerRef.current) {
        clearInterval(elapsedTimerRef.current);
        elapsedTimerRef.current = null;
      }
    }
  };

  const exampleQueries = [
    'What is the current price of Google stock?',
    'Tell me about Python programming language',
    'What is 1234 * 5678?',
    'Compare the stock prices of Google and Apple'
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <Brain size={32} />
          <h1>ReAct Agent Dashboard</h1>
          <p>L32: Hands-On ReAct Agent Build</p>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <div className="metrics-dashboard">
            <div className="metric-card">
              <Wrench size={24} />
              <div>
                <span className="metric-value">{metrics.tools_loaded}</span>
                <span className="metric-label">Tools Loaded</span>
              </div>
            </div>
            <div className="metric-card">
              <Database size={24} />
              <div>
                <span className="metric-value">{metrics.active_sessions}</span>
                <span className="metric-label">Active Sessions</span>
              </div>
            </div>
            <div className="metric-card">
              <Zap size={24} />
              <div>
                <span className="metric-value">{metrics.total_queries}</span>
                <span className="metric-label">Total Queries</span>
              </div>
            </div>
            <div className="metric-card">
              <Activity size={24} />
              <div>
                <span className={`metric-value status-${metrics.status}`}>{metrics.status}</span>
                <span className="metric-label">API Status</span>
              </div>
            </div>
          </div>

          <div className="query-section">
            <div className="query-input-container">
              <textarea
                className="query-input"
                placeholder="Enter your query (e.g., 'What is the current price of Google stock?')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={3}
              />
              <div className="button-group">
                <button 
                  className="btn btn-primary" 
                  onClick={executeQuery}
                  disabled={loading || !query.trim()}
                >
                  {loading ? <Loader className="spin" size={20} /> : <Send size={20} />}
                  {loading ? `Processing... ${elapsedSeconds}s` : 'Execute'}
                </button>
                <button 
                  className="btn btn-secondary" 
                  onClick={fetchTools}
                >
                  <Wrench size={20} />
                  {showTools ? 'Hide Tools' : 'View Tools'}
                </button>
              </div>
            </div>

            {loading && (
              <p className="loading-hint">Queries may take 30–90 seconds. Retrying automatically on timeout.</p>
            )}
            <div className="examples">
              <p className="examples-title">Example Queries:</p>
              <div className="example-chips">
                {exampleQueries.map((ex, idx) => (
                  <button 
                    key={idx}
                    className="chip"
                    onClick={() => setQuery(ex)}
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {showTools && (
            <div className="tools-panel">
              <h3>Available Tools ({tools.length})</h3>
              <div className="tools-grid">
                {tools.map((tool, idx) => (
                  <div key={idx} className="tool-card">
                    <h4>{tool.name}</h4>
                    <p>{tool.description}</p>
                    <div className="tool-params">
                      <strong>Parameters:</strong>
                      <ul>
                        {Object.entries(tool.parameters).map(([key, value]) => (
                          <li key={key}>
                            <code>{key}</code>: {value.description}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result && (
            <div className="result-section">
              <div className={`result-header ${result.success ? 'success' : 'error'}`}>
                {result.success ? <CheckCircle size={24} /> : <XCircle size={24} />}
                <h3>{result.success ? 'Task Completed' : 'Task Failed'}</h3>
                <span className="iterations">
                  {result.iterations_used} iterations
                </span>
              </div>

              {(() => {
                const hasApiKeyError = !result.success && (
                  /API_KEY|api key expired|API key expired|invalid.*key/i.test(result.error || '') ||
                  (result.reasoning_trace || []).some(s => /API_KEY|api key expired|API key expired/i.test(s.observation || ''))
                );
                const hasRateLimitError = !result.success && (
                  /429|quota|rate limit|exceeded.*quota/i.test(result.error || '') ||
                  (result.reasoning_trace || []).some(s => /429|quota exceeded|rate limit/i.test(s.observation || ''))
                );
                return (
                  <>
                    {hasApiKeyError && (
                      <div className="api-key-alert">
                        <strong>API key expired or invalid.</strong> Get a new key at{' '}
                        <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a>, add{' '}
                        <code>GEMINI_API_KEY=your_key</code> to <code>.env</code>, then restart with <code>./scripts/stop.sh && ./scripts/start.sh</code>.
                      </div>
                    )}
                    {hasRateLimitError && (
                      <div className="api-key-alert" style={{ borderColor: '#2196f3', background: '#e3f2fd', color: '#1565c0' }}>
                        <strong>API quota exhausted.</strong> Simple queries (stock, calculator, Wikipedia) use direct tools with no API limit. For AI reasoning, enable billing at{' '}
                        <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer">Gemini billing</a> for higher limits.
                      </div>
                    )}
                  </>
                );
              })()}

              <div className="final-answer">
                <h4>Final Answer:</h4>
                <p>{result.result || result.error}</p>
              </div>

              {result.reasoning_trace && result.reasoning_trace.length > 0 && (
                <div className="reasoning-trace">
                  <h4>Reasoning Trace:</h4>
                  {result.reasoning_trace.map((step, idx) => (
                    <div key={idx} className="reasoning-step">
                      <div className="step-header">
                        <span className="step-number">Step {step.step_number}</span>
                        <span className="step-action">{step.action}</span>
                      </div>
                      <div className="step-content">
                        <div className="step-item">
                          <strong>💭 Thought:</strong>
                          <p>{step.thought}</p>
                        </div>
                        {step.action_input && Object.keys(step.action_input).length > 0 && (
                          <div className="step-item">
                            <strong>⚙️ Action Input:</strong>
                            <pre>{JSON.stringify(step.action_input, null, 2)}</pre>
                          </div>
                        )}
                        <div className="step-item">
                          <strong>👁️ Observation:</strong>
                          <p>{step.observation}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
