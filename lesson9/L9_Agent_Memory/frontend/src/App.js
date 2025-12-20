import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [sessionId, setSessionId] = useState('session_001');
  const [userId, setUserId] = useState('user_001');
  const [message, setMessage] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [shortTermMemory, setShortTermMemory] = useState([]);
  const [longTermResults, setLongTermResults] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [factInput, setFactInput] = useState('');

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/metrics`);
      setMetrics(res.data.metrics);
    } catch (err) {
      console.error('Metrics error:', err);
    }
  };

  const storeMessage = async () => {
    if (!message.trim()) return;
    
    try {
      await axios.post(`${API_BASE}/api/store`, {
        session_id: sessionId,
        role: 'user',
        content: message,
        metadata: { source: 'dashboard' }
      });
      
      setMessage('');
      recallSession();
      loadMetrics();
    } catch (err) {
      alert('Error storing message: ' + err.message);
    }
  };

  const recallSession = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/recall/${sessionId}`);
      setShortTermMemory(res.data.messages);
    } catch (err) {
      console.error('Recall error:', err);
    }
  };

  const storeLongTermFact = async () => {
    if (!factInput.trim()) return;
    
    try {
      await axios.post(`${API_BASE}/api/longterm/store`, {
        user_id: userId,
        fact: { content: factInput, type: 'user_preference' }
      });
      
      setFactInput('');
      alert('Fact stored in long-term memory!');
      loadMetrics();
    } catch (err) {
      alert('Error storing fact: ' + err.message);
    }
  };

  const searchLongTerm = async () => {
    try {
      const keywords = searchQuery.split(' ').filter(k => k.length > 0);
      const res = await axios.post(`${API_BASE}/api/longterm/search`, {
        user_id: userId,
        keywords: keywords.length > 0 ? keywords : null,
        limit: 5
      });
      
      setLongTermResults(res.data.results);
      loadMetrics();
    } catch (err) {
      alert('Search error: ' + err.message);
    }
  };

  const clearSession = async () => {
    try {
      await axios.delete(`${API_BASE}/api/session/${sessionId}`);
      setShortTermMemory([]);
      alert('Session cleared!');
      loadMetrics();
    } catch (err) {
      alert('Clear error: ' + err.message);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🧠 Agent Memory System</h1>
        <p>L9: Dual-Tier Memory Architecture</p>
      </header>

      <div className="container">
        {/* Metrics Panel */}
        <div className="metrics-panel">
          <h2>📊 Memory Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Short-Term Hits</span>
              <span className="metric-value">{metrics.short_term_hits || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Long-Term Hits</span>
              <span className="metric-value">{metrics.long_term_hits || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Active Sessions</span>
              <span className="metric-value">{metrics.active_sessions || 0}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Total Stores</span>
              <span className="metric-value">{metrics.stores || 0}</span>
            </div>
          </div>
        </div>

        <div className="two-column">
          {/* Short-Term Memory */}
          <div className="panel">
            <h2>💬 Short-Term Memory</h2>
            <div className="config">
              <label>Session ID:</label>
              <input 
                type="text" 
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="session_001"
              />
            </div>
            
            <div className="message-input">
              <input 
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && storeMessage()}
                placeholder="Enter a message..."
              />
              <button onClick={storeMessage}>Store</button>
              <button onClick={recallSession} className="secondary">Recall</button>
              <button onClick={clearSession} className="danger">Clear</button>
            </div>

            <div className="memory-list">
              {shortTermMemory.length === 0 ? (
                <p className="empty">No messages in session</p>
              ) : (
                shortTermMemory.map((msg, idx) => (
                  <div key={idx} className="memory-item">
                    <span className="role">{msg.role}</span>
                    <span className="content">{msg.content}</span>
                    <span className="timestamp">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Long-Term Memory */}
          <div className="panel">
            <h2>💾 Long-Term Memory</h2>
            <div className="config">
              <label>User ID:</label>
              <input 
                type="text" 
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="user_001"
              />
            </div>

            <div className="fact-input">
              <input 
                type="text"
                value={factInput}
                onChange={(e) => setFactInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && storeLongTermFact()}
                placeholder="Store a fact (e.g., 'Likes pizza')"
              />
              <button onClick={storeLongTermFact}>Store Fact</button>
            </div>

            <div className="search-input">
              <input 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchLongTerm()}
                placeholder="Search keywords..."
              />
              <button onClick={searchLongTerm}>Search</button>
            </div>

            <div className="memory-list">
              {longTermResults.length === 0 ? (
                <p className="empty">No results found</p>
              ) : (
                longTermResults.map((entry, idx) => (
                  <div key={idx} className="memory-item">
                    <span className="content">{JSON.stringify(entry.fact)}</span>
                    <span className="timestamp">{new Date(entry.timestamp).toLocaleDateString()}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
