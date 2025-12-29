import React, { useState, useEffect } from 'react';
import { MessageSquare, Database, History, BarChart3 } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [sessionId, setSessionId] = useState('');
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);
  const [stateInfo, setStateInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Generate session ID on mount
    const id = `session_${Date.now()}`;
    setSessionId(id);
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message })
      });

      const data = await response.json();
      
      setConversation(prev => [...prev, {
        user: message,
        agent: data.response,
        tokens: data.tokens_used,
        version: data.state_version
      }]);

      setMessage('');
      
      // Fetch updated state
      await fetchState();
    } catch (error) {
      console.error('Chat error:', error);
    }
    setLoading(false);
  };

  const fetchState = async () => {
    try {
      const response = await fetch(`${API_URL}/api/state/${sessionId}`);
      const data = await response.json();
      setStateInfo(data);
    } catch (error) {
      console.error('State fetch error:', error);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 L14: State Management Dashboard</h1>
        <p>Session: {sessionId}</p>
      </header>

      <div className="main-grid">
        {/* Chat Interface */}
        <div className="panel chat-panel">
          <div className="panel-header">
            <MessageSquare size={20} />
            <h2>Conversation</h2>
          </div>
          
          <div className="conversation">
            {conversation.map((turn, idx) => (
              <div key={idx} className="turn">
                <div className="message user-message">
                  <strong>You:</strong> {turn.user}
                </div>
                <div className="message agent-message">
                  <strong>Agent:</strong> {turn.agent}
                  <div className="message-meta">
                    v{turn.version} • {turn.tokens} tokens
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="input-area">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type a message..."
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading}>
              {loading ? 'Processing...' : 'Send'}
            </button>
          </div>
        </div>

        {/* State Inspector */}
        <div className="panel state-panel">
          <div className="panel-header">
            <Database size={20} />
            <h2>State Inspector</h2>
          </div>

          {stateInfo && (
            <div className="state-info">
              <div className="stat-card">
                <div className="stat-label">Version</div>
                <div className="stat-value">{stateInfo.version}</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Status</div>
                <div className="stat-value">{stateInfo.state_status}</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Total Turns</div>
                <div className="stat-value">{stateInfo.total_turns}</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Total Tokens</div>
                <div className="stat-value">{stateInfo.total_tokens}</div>
              </div>

              <div className="history-section">
                <h3><History size={16} /> Conversation History</h3>
                <div className="history-list">
                  {stateInfo.conversation_history?.slice(-5).map((turn, idx) => (
                    <div key={idx} className="history-item">
                      <div className="history-turn">Turn {turn.turn_id}</div>
                      <div className="history-tokens">{turn.tokens_used} tokens</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Metrics */}
        <div className="panel metrics-panel">
          <div className="panel-header">
            <BarChart3 size={20} />
            <h2>Metrics</h2>
          </div>

          <div className="metrics">
            <div className="metric">
              <div className="metric-label">Average Tokens/Turn</div>
              <div className="metric-value">
                {stateInfo && stateInfo.total_turns > 0 
                  ? Math.round(stateInfo.total_tokens / stateInfo.total_turns)
                  : 0}
              </div>
            </div>

            <div className="metric">
              <div className="metric-label">State Persistence</div>
              <div className="metric-value">✓ Active</div>
            </div>

            <div className="metric">
              <div className="metric-label">Cache Status</div>
              <div className="metric-value">Redis Hot</div>
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>L14: State Management • VAIA Curriculum • Enterprise-Grade Persistence</p>
      </footer>
    </div>
  );
}

export default App;
