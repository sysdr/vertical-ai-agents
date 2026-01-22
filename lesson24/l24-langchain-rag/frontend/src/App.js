import React, { useState, useEffect } from 'react';
import axios from 'axios';
import QueryInterface from './components/QueryInterface';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stats`);
      setStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🔗 L24: Modular RAG with LangChain</h1>
        <p>Enterprise-grade retrieval chains with flexible orchestration</p>
      </header>

      {loading ? (
        <div className="loading">Loading system stats...</div>
      ) : (
        <div className="stats-bar">
          <div className="stat">
            <span className="stat-label">Documents</span>
            <span className="stat-value">{stats?.total_documents || 0}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Model</span>
            <span className="stat-value">{stats?.model || 'N/A'}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Retriever</span>
            <span className="stat-value">{stats?.retriever_type || 'N/A'}</span>
          </div>
        </div>
      )}

      <QueryInterface apiUrl={API_URL} />
    </div>
  );
}

export default App;
