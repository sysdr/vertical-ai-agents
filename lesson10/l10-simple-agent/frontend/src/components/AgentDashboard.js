import React, { useState, useEffect } from 'react';
import './AgentDashboard.css';

const API_URL = 'http://localhost:8000';

function AgentDashboard() {
  const [message, setMessage] = useState('');
  const [goal, setGoal] = useState('');
  const [agentState, setAgentState] = useState(null);
  const [responses, setResponses] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAgentState();
    fetchDecisions();
    const interval = setInterval(() => {
      fetchAgentState();
      fetchDecisions();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchAgentState = async () => {
    try {
      const res = await fetch(`${API_URL}/agent/state`);
      const data = await res.json();
      setAgentState(data);
    } catch (err) {
      console.error('Failed to fetch agent state:', err);
    }
  };

  const fetchDecisions = async () => {
    try {
      const res = await fetch(`${API_URL}/agent/decisions`);
      const data = await res.json();
      setDecisions(data.decisions || []);
    } catch (err) {
      console.error('Failed to fetch decisions:', err);
    }
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/agent/act`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: message,
          goal: goal || null
        })
      });

      const data = await res.json();
      setResponses(prev => [...prev, {
        message,
        goal,
        response: data,
        timestamp: new Date().toLocaleTimeString()
      }]);

      setMessage('');
      fetchAgentState();
      fetchDecisions();
    } catch (err) {
      console.error('Failed to send message:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStateColor = (state) => {
    const colors = {
      'IDLE': '#95a5a6',
      'THINKING': '#3498db',
      'ACTING': '#f39c12',
      'EVALUATING': '#9b59b6',
      'COMPLETE': '#27ae60',
      'FAILED': '#e74c3c',
      'READY': '#16a085'
    };
    return colors[state] || '#95a5a6';
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🤖 L10: Simple Agent</h1>
        <p>Autonomous agent with memory & goal-seeking</p>
      </header>

      <div className="dashboard-grid">
        {/* Agent State */}
        <div className="card state-card">
          <h2>Agent State</h2>
          {agentState && (
            <div className="state-display">
              <div className="state-badge" style={{ backgroundColor: getStateColor(agentState.state) }}>
                {agentState.state}
              </div>
              <div className="state-details">
                <div className="state-item">
                  <span className="label">Current Goal:</span>
                  <span className="value">{agentState.current_goal || 'None'}</span>
                </div>
                <div className="state-item">
                  <span className="label">Attempt:</span>
                  <span className="value">{agentState.attempt_count}/5</span>
                </div>
                <div className="state-item">
                  <span className="label">Conversations:</span>
                  <span className="value">{agentState.conversation_length}</span>
                </div>
                <div className="state-item">
                  <span className="label">Decisions Made:</span>
                  <span className="value">{agentState.decision_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="card input-card">
          <h2>Send Message</h2>
          <div className="input-group">
            <input
              type="text"
              placeholder="Enter goal (optional)"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="goal-input"
            />
            <textarea
              placeholder="Enter your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows="4"
              className="message-input"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !message.trim()}
              className="send-button"
            >
              {loading ? 'Processing...' : 'Send to Agent'}
            </button>
          </div>
        </div>

        {/* Responses */}
        <div className="card responses-card">
          <h2>Agent Responses</h2>
          <div className="responses-list">
            {responses.slice().reverse().map((item, idx) => (
              <div key={idx} className="response-item">
                <div className="response-header">
                  <span className="timestamp">{item.timestamp}</span>
                  {item.goal && <span className="goal-badge">{item.goal}</span>}
                </div>
                <div className="response-message">
                  <strong>You:</strong> {item.message}
                </div>
                <div className="response-action">
                  <strong>Agent Action:</strong> {item.response.action}
                </div>
                <div className="response-reasoning">
                  <strong>Reasoning:</strong> {item.response.reasoning}
                </div>
                {item.response.progress && (
                  <div className="response-progress">
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${item.response.progress.progress_percentage}%` }}
                      />
                    </div>
                    <span>{item.response.progress.progress_percentage}% complete</span>
                  </div>
                )}
              </div>
            ))}
            {responses.length === 0 && (
              <div className="empty-state">No responses yet. Send a message to start!</div>
            )}
          </div>
        </div>

        {/* Decision Log */}
        <div className="card decisions-card">
          <h2>Decision Log (Last 10)</h2>
          <div className="decisions-list">
            {decisions.slice().reverse().map((decision, idx) => (
              <div key={idx} className="decision-item">
                <div className="decision-header">
                  <span className="attempt-badge">Attempt {decision.attempt}</span>
                  <span className="goal-text">{decision.goal}</span>
                </div>
                <div className="decision-action">{decision.action}</div>
                <div className="decision-states">
                  {decision.state_transitions.map((state, i) => (
                    <span key={i} className="state-chip" style={{ backgroundColor: getStateColor(state) }}>
                      {state}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {decisions.length === 0 && (
              <div className="empty-state">No decisions logged yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentDashboard;
