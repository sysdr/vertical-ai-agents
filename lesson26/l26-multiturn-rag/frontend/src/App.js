import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionInfo, setSessionInfo] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    createNewSession();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (sessionId) {
      const interval = setInterval(() => {
        fetchSessionInfo();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const createNewSession = async () => {
    try {
      const response = await axios.post(`${API_BASE}/session/new`);
      setSessionId(response.data.session_id);
      setMessages([]);
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const fetchSessionInfo = async () => {
    if (!sessionId) return;
    
    try {
      const response = await axios.get(`${API_BASE}/session/${sessionId}/info`);
      setSessionInfo(response.data);
    } catch (error) {
      // 404 = session not stored yet (no messages); show default instead of logging error
      if (error.response?.status === 404) {
        setSessionInfo({
          session_id: sessionId,
          message_count: 0,
          total_tokens: 0,
          state: 'NEW',
          created_at: new Date().toISOString(),
          last_activity: new Date().toISOString()
        });
      } else {
        console.error('Failed to fetch session info:', error);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        session_id: sessionId,
        query: input
      });

      const data = response.data;

      if (data.is_ambiguous) {
        // Show clarification questions
        const clarificationMessage = {
          role: 'assistant',
          content: data.response,
          clarifications: data.clarification_questions,
          ambiguity_score: data.ambiguity_score,
          isAmbiguous: true,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, clarificationMessage]);
      } else {
        // Show normal response
        const assistantMessage = {
          role: 'assistant',
          content: data.response,
          documents: data.retrieved_documents,
          ambiguity_score: data.ambiguity_score,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }

      await fetchSessionInfo();
    } catch (error) {
      console.error('Query failed:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Error: ' + (error.response?.data?.detail || error.message),
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClarificationClick = (question) => {
    setInput(question);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🤖 L26: Multi-Turn RAG System</h1>
        <p>Conversational AI with Ambiguity Handling</p>
      </header>

      <div className="main-container">
        {/* Session Info Sidebar */}
        <div className="sidebar">
          <div className="info-card">
            <h3>Session Info</h3>
            {sessionInfo ? (
              <div className="info-content">
                <div className="info-row">
                  <span className="info-label">Messages:</span>
                  <span className="info-value">{sessionInfo.message_count}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Tokens:</span>
                  <span className="info-value">{sessionInfo.total_tokens}</span>
                </div>
                <div className="info-row">
                  <span className="info-label">State:</span>
                  <span className={`state-badge ${sessionInfo.state.toLowerCase()}`}>
                    {sessionInfo.state}
                  </span>
                </div>
                <div className="info-row">
                  <span className="info-label">Session ID:</span>
                  <span className="info-value session-id">
                    {sessionId.substring(0, 8)}...
                  </span>
                </div>
              </div>
            ) : (
              <p className="loading-text">Loading...</p>
            )}
          </div>

          <button className="new-session-btn" onClick={createNewSession}>
            🔄 New Session
          </button>

          <div className="info-card">
            <h3>Features</h3>
            <ul className="feature-list">
              <li>✅ Conversation Memory</li>
              <li>✅ Ambiguity Detection</li>
              <li>✅ Context Expansion</li>
              <li>✅ Clarification Questions</li>
              <li>✅ Multi-Turn Retrieval</li>
            </ul>
          </div>
        </div>

        {/* Chat Container */}
        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <h2>Start a Conversation</h2>
                <p>Try asking about pricing, support, or API features</p>
                <div className="sample-queries">
                  <button onClick={() => setInput("What are your pricing plans?")}>
                    💰 Pricing Plans
                  </button>
                  <button onClick={() => setInput("Tell me about support")}>
                    🎧 Support Info
                  </button>
                  <button onClick={() => setInput("How do I authenticate?")}>
                    🔐 API Authentication
                  </button>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-header">
                  <span className="message-role">
                    {msg.role === 'user' ? '👤 You' : 
                     msg.role === 'assistant' ? '🤖 Assistant' : 
                     '⚠️ System'}
                  </span>
                  {msg.ambiguity_score !== undefined && (
                    <span className="ambiguity-badge">
                      Ambiguity: {(msg.ambiguity_score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                
                <div className="message-content">
                  {msg.content}
                </div>

                {msg.isAmbiguous && msg.clarifications && (
                  <div className="clarifications">
                    <p className="clarifications-label">Please clarify:</p>
                    {msg.clarifications.map((q, qIdx) => (
                      <button
                        key={qIdx}
                        className="clarification-btn"
                        onClick={() => handleClarificationClick(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}

                {msg.documents && msg.documents.length > 0 && (
                  <div className="documents">
                    <p className="documents-label">
                      📚 Retrieved {msg.documents.length} document(s)
                    </p>
                    <div className="documents-list">
                      {msg.documents.map((doc, docIdx) => (
                        <div key={docIdx} className="document-item">
                          <div className="document-header">
                            <span className="document-id">Doc {docIdx + 1}</span>
                            <span className="document-distance">
                              Score: {(1 - doc.distance).toFixed(3)}
                            </span>
                          </div>
                          <p className="document-text">{doc.text.substring(0, 150)}...</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="message assistant loading">
                <div className="message-header">
                  <span className="message-role">🤖 Assistant</span>
                </div>
                <div className="message-content">
                  <span className="typing-indicator">
                    <span></span><span></span><span></span>
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form className="input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading}
              className="input-field"
            />
            <button 
              type="submit" 
              disabled={loading || !input.trim()}
              className="send-btn"
            >
              {loading ? '⏳' : '📤'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
