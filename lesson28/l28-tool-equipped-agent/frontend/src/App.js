import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [quickMode, setQuickMode] = useState(false); // Skips evaluation for faster RAG
  const [result, setResult] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/stats`);
      setStats(res.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        query: query,
        conversation_id: conversationId,
        include_metrics: !quickMode
      });

      const data = response.data;
      setResult(data);
      setConversationId(data.conversation_id);
      
      setHistory(prev => [...prev, {
        query: query,
        response: data.response,
        intent: data.intent_type,
        tool: data.tool_used,
        metrics: data.metrics
      }]);
      
      setQuery('');
      fetchStats();
    } catch (error) {
      console.error('Error:', error);
      let msg = error.response?.data?.detail || error.message || 'Error processing query';
      if (typeof msg !== 'string') msg = JSON.stringify(msg);
      // Show user-friendly message for API key errors
      const apiKeyErrorPatterns = ['API_KEY_INVALID', 'api key expired', 'api key invalid', 'renew the api key'];
      if (apiKeyErrorPatterns.some(p => msg.toLowerCase().includes(p.toLowerCase()))) {
        msg = 'Invalid or expired Gemini API key. Set GEMINI_API_KEY in backend/.env and restart the backend.';
      } else if (msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('429') || msg.toLowerCase().includes('rate limit')) {
        msg = 'API quota exceeded. Wait a few minutes and try again. See https://ai.google.dev/gemini-api/docs/rate-limits';
      }
      alert(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>🤖 L28: Tool-Equipped Agent</h1>
        <p>RAG + Tools + Evaluation</p>
      </header>

      <div className="container">
        {stats && (
          <div className="dashboard-section">
            <h3>📊 Dashboard Metrics</h3>
            {stats.total_queries === 0 && (
              <p className="dashboard-hint">Try sample questions below. Calculator, weather, and database work without API quota.</p>
            )}
            <div className="dashboard-grid">
              <div className="dashboard-card">
                <label>Total Queries</label>
                <div className="dashboard-value">{stats.total_queries}</div>
              </div>
              <div className="dashboard-card">
                <label>RAG Queries</label>
                <div className="dashboard-value">{stats.rag_count}</div>
              </div>
              <div className="dashboard-card">
                <label>Tool Queries</label>
                <div className="dashboard-value">{stats.tool_count}</div>
              </div>
              <div className="dashboard-card">
                <label>Hybrid Queries</label>
                <div className="dashboard-value">{stats.hybrid_count}</div>
              </div>
            </div>
            {stats.metrics_count > 0 && stats.metrics_avg && (
              <div className="dashboard-metrics">
                <h4>Average Evaluation Scores ({stats.metrics_count} RAG queries)</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <label>Groundedness</label>
                    <div className="score">{(stats.metrics_avg.groundedness * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Relevance</label>
                    <div className="score">{(stats.metrics_avg.relevance * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Completeness</label>
                    <div className="score">{(stats.metrics_avg.completeness * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Overall</label>
                    <div className="score overall">{(stats.metrics_avg.overall * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            )}
            {stats.tools_used && Object.keys(stats.tools_used).length > 0 && (
              <div className="tools-used">
                <h4>Tools Used</h4>
                <div className="tools-list">
                  {Object.entries(stats.tools_used).map(([name, count]) => (
                    <span key={name} className="tool-badge">{name}: {count}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        <div className="sample-questions">
          <h4>Sample questions (click to try):</h4>
          <div className="sample-grid">
            <button type="button" className="sample-btn" onClick={() => setQuery('Calculate 2+2')}>Calculate 2+2</button>
            <button type="button" className="sample-btn" onClick={() => setQuery("What's the weather in Tokyo?")}>Weather in Tokyo</button>
            <button type="button" className="sample-btn" onClick={() => setQuery('List users from the database')}>List users</button>
            <button type="button" className="sample-btn" onClick={() => setQuery('What is Python?')}>What is Python? (RAG)</button>
            <button type="button" className="sample-btn" onClick={() => setQuery('What is FastAPI and calculate 2^10')}>FastAPI + calc (Hybrid)</button>
            <button type="button" className="sample-btn" onClick={() => setQuery('Calculate 15 * 27')}>15 × 27</button>
            <button type="button" className="sample-btn" onClick={() => setQuery('Show products')}>Show products</button>
          </div>
        </div>
        <div className="query-section">
          <form onSubmit={handleSubmit}>
            <label className="quick-mode-toggle">
              <input type="checkbox" checked={quickMode} onChange={(e) => setQuickMode(e.target.checked)} />
              Quick response (skips evaluation, faster RAG)
            </label>
            <div className="query-row">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question, request calculation, or check weather..."
                disabled={loading}
              />
              <button type="submit" disabled={loading}>
                {loading ? 'Processing...' : 'Send'}
              </button>
            </div>
          </form>
        </div>

        {result && (
          <div className="result-section">
            <div className="result-header">
              <span className="badge">{result.intent_type}</span>
              {result.tool_used && (
                <span className="badge tool">Tool: {result.tool_used}</span>
              )}
            </div>
            
            <div className="response">
              <h3>Response:</h3>
              <p>{result.response}</p>
            </div>

            {result.sources && result.sources.length > 0 && (
              <div className="sources">
                <h4>Sources:</h4>
                <ul>
                  {result.sources.map((src, idx) => (
                    <li key={idx}>{src}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.metrics && (
              <div className="metrics">
                <h4>Evaluation Metrics:</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <label>Groundedness</label>
                    <div className="score">{(result.metrics.groundedness_score * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Relevance</label>
                    <div className="score">{(result.metrics.relevance_score * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Completeness</label>
                    <div className="score">{(result.metrics.completeness_score * 100).toFixed(0)}%</div>
                  </div>
                  <div className="metric">
                    <label>Overall</label>
                    <div className="score overall">{(result.metrics.overall_score * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {history.length > 0 && (
          <div className="history">
            <h3>Conversation History</h3>
            {history.map((item, idx) => (
              <div key={idx} className="history-item">
                <div className="history-query">
                  <strong>Q:</strong> {item.query}
                  <span className="history-badge">{item.intent}</span>
                </div>
                <div className="history-response">
                  <strong>A:</strong> {item.response.substring(0, 150)}...
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
