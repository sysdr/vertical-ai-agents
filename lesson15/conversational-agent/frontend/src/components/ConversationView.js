import React, { useMemo } from 'react';
import './ConversationView.css';

function ConversationView({ history, goals }) {
  // Ensure goals is always an array and force re-render when it changes
  const goalsArray = useMemo(() => {
    const arr = Array.isArray(goals) ? goals : [];
    // Debug: log when goals change
    if (arr.length > 0) {
      console.log('ConversationView: Goals updated', arr);
    }
    return arr;
  }, [goals]);
  
  return (
    <div className="conversation-container">
      <div className="goals-panel">
        <h3>Active Goals</h3>
        {goalsArray.length === 0 ? (
          <div className="empty">
            <p>No goals set.</p>
            <p style={{ fontSize: '0.9em', marginTop: '8px', color: '#666' }}>
              Use <code>/goal &lt;description&gt;</code> to add one.
            </p>
            <p style={{ fontSize: '0.85em', marginTop: '8px', color: '#999', fontStyle: 'italic' }}>
              Example: <code>/goal Learn Python programming</code>
            </p>
          </div>
        ) : (
          <div className="goals-list">
            {goalsArray.map((goal, idx) => (
              <div key={idx} className={`goal ${goal.completed ? 'completed' : 'active'}`}>
                <div className="goal-icon">{goal.completed ? '✓' : '○'}</div>
                <div className="goal-content">
                  <div className="goal-description">{goal.description}</div>
                  <div className="goal-meta">
                    {goal.created_at ? new Date(goal.created_at).toLocaleString() : 'Just now'}
                    {goal.completed && <span style={{ marginLeft: '8px', color: '#28a745' }}>• Completed</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="messages-panel">
        <h3>Conversation History</h3>
        {history.length === 0 ? (
          <p className="empty">No messages yet. Start the conversation!</p>
        ) : (
          <div className="messages-list">
            {history.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  <div className="message-meta">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                    {msg.token_count > 0 && ` • ${msg.token_count} tokens`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ConversationView;
