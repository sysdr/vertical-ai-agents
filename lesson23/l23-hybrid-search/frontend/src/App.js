import React, { useState, useEffect } from 'react';
import axios from 'axios';
import SearchInterface from './components/SearchInterface';
import DocumentManager from './components/DocumentManager';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [activeTab, setActiveTab] = useState('search');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 L23: Hybrid Search & Filtering</h1>
        <p className="subtitle">Vector + Keyword Search with Metadata Filtering & RRF Fusion</p>
        {stats && (
          <div className="stats-bar">
            <span>📚 {stats.total_documents} documents</span>
            <span>🏷️ {stats.sample_metadata_fields.length} metadata fields</span>
          </div>
        )}
      </header>

      <div className="tab-nav">
        <button 
          className={activeTab === 'search' ? 'active' : ''}
          onClick={() => setActiveTab('search')}
        >
          Search
        </button>
        <button 
          className={activeTab === 'manage' ? 'active' : ''}
          onClick={() => setActiveTab('manage')}
        >
          Manage Documents
        </button>
      </div>

      <main className="app-main">
        {activeTab === 'search' ? (
          <SearchInterface apiUrl={API_URL} />
        ) : (
          <DocumentManager apiUrl={API_URL} onUpdate={fetchStats} />
        )}
      </main>

      <footer className="app-footer">
        <p>Advanced Architectures for Vertical AI Agents - Lesson 23 of 90</p>
        <p>Build on L22 (Reranking) | Enables L24 (LangChain Integration)</p>
      </footer>
    </div>
  );
}

export default App;
