import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ConversationView from './components/ConversationView';

const API_URL = 'http://localhost:8000';

function App() {
  const [userId, setUserId] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [message, setMessage] = useState('');
  const [history, setHistory] = useState([]);
  const [goals, setGoals] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(false);

  const createConversation = async () => {
    if (!userId.trim()) return;
    try {
      const res = await axios.post(`${API_URL}/conversations`, { user_id: userId });
      const newConvId = res.data.conversation_id;
      setConversationId(newConvId);
      setHistory([]);
      setGoals([]);
      setStats({ state: 'initializing', total_messages: 0, active_goals: 0, total_tokens: 0 });
      // Load initial history (should be empty but ensures goals are loaded)
      await loadHistory();
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  const sendMessage = async () => {
    if (!message.trim() || !conversationId) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/messages`, {
        conversation_id: conversationId,
        message: message
      });
      // Set stats from API response first (this has the correct active_goals count)
      const apiStats = res.data;
      setStats(apiStats);
      
      // Immediately reload history to get updated goals - no delay needed
      // The backend saves synchronously, so we can reload right away
      await loadHistory();
      
      setMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
      // Show user-friendly error message
      if (error.response?.data?.detail) {
        alert(`Error: ${error.response.data.detail}`);
      } else if (error.message) {
        alert(`Error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    if (!conversationId) return;
    try {
      const res = await axios.get(`${API_URL}/conversations/${conversationId}/history`);
      const messages = res.data.messages || [];
      const goals = res.data.goals || [];
      
      // Always set goals as an array - create new array reference to trigger re-render
      const goalsArray = Array.isArray(goals) ? goals : [];
      
      // Create new array references to ensure React detects the change
      setHistory([...messages]);
      setGoals([...goalsArray]); // New array reference triggers ConversationView re-render
      
      // Calculate active goals from goals array (this is the source of truth)
      const totalTokens = messages.reduce((sum, msg) => sum + (msg.token_count || 0), 0);
      const activeGoalsCount = goalsArray.filter(g => !g.completed).length;
      
      setStats(prev => ({
        ...prev,
        total_messages: messages.length,
        // Always use calculated value from goals array (source of truth)
        active_goals: activeGoalsCount,
        total_tokens: totalTokens,
        state: res.data.state || prev.state
      }));
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  useEffect(() => {
    if (conversationId) {
      loadHistory();
      // Set up periodic refresh to catch goal updates during conversation
      const interval = setInterval(() => {
        loadHistory();
      }, 2000); // Refresh every 2 seconds to catch goal completion updates
      
      return () => clearInterval(interval);
    }
  }, [conversationId]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Conversational Agent Dashboard</h1>
      </header>
      
      <div className="container">
        {!conversationId ? (
          <div className="setup-panel">
            <h2>Start New Conversation</h2>
            <input
              type="text"
              placeholder="Enter User ID"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && createConversation()}
            />
            <button onClick={createConversation} disabled={!userId.trim()}>
              Create Conversation
            </button>
          </div>
        ) : (
          <>
            <div className="stats-panel">
              <div className="stat">
                <span className="label">State:</span>
                <span className="value">{stats.state || 'N/A'}</span>
              </div>
              <div className="stat">
                <span className="label">Messages:</span>
                <span className="value">{stats.total_messages || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Active Goals:</span>
                <span className="value">{typeof stats.active_goals === 'number' ? stats.active_goals : (goals ? goals.filter(g => !g.completed).length : 0)}</span>
              </div>
              <div className="stat">
                <span className="label">Tokens:</span>
                <span className="value">{stats.total_tokens || 0}</span>
              </div>
            </div>

            <ConversationView history={history} goals={goals} />

            <div className="input-panel">
              <input
                type="text"
                placeholder="Type a message... (use /goal <text> to set a goal)"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !loading && sendMessage()}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !message.trim()}>
                {loading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
