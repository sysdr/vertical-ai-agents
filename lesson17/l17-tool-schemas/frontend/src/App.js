import React, { useState, useEffect } from 'react';
import ToolsList from './components/ToolsList';
import QueryInterface from './components/QueryInterface';
import ValidationTester from './components/ValidationTester';
import MetricsDashboard from './components/MetricsDashboard';
import './App.css';

function App() {
  const [tools, setTools] = useState([]);
  const [activeTab, setActiveTab] = useState('query');
  const [status, setStatus] = useState({ connected: false, toolsCount: 0 });

  useEffect(() => {
    loadTools();
    checkHealth();
  }, []);

  const loadTools = async () => {
    try {
      const response = await fetch('http://localhost:8000/tools');
      const data = await response.json();
      setTools(data.tools);
      setStatus(prev => ({ ...prev, toolsCount: data.count }));
    } catch (error) {
      console.error('Failed to load tools:', error);
    }
  };

  const checkHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      setStatus(prev => ({ ...prev, connected: data.status === 'healthy' }));
    } catch (error) {
      setStatus(prev => ({ ...prev, connected: false }));
    }
  };

  return (
    <div className="App">
      <header className="header">
        <div className="header-content">
          <h1>🛡️ L17: Tool Schema Validator</h1>
          <p className="subtitle">Production-Grade Pydantic Schema Validation for LLM Tools</p>
          <div className="status-bar">
            <span className={`status-indicator ${status.connected ? 'connected' : 'disconnected'}`}>
              {status.connected ? '✓ Connected' : '✗ Disconnected'}
            </span>
            <span className="tools-count">{status.toolsCount} tools registered</span>
          </div>
        </div>
      </header>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'query' ? 'active' : ''}`}
          onClick={() => setActiveTab('query')}
        >
          Query Interface
        </button>
        <button 
          className={`tab ${activeTab === 'validate' ? 'active' : ''}`}
          onClick={() => setActiveTab('validate')}
        >
          Validation Tester
        </button>
        <button 
          className={`tab ${activeTab === 'tools' ? 'active' : ''}`}
          onClick={() => setActiveTab('tools')}
        >
          Schema Browser
        </button>
        <button 
          className={`tab ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics Dashboard
        </button>
      </div>

      <main className="main-content">
        {activeTab === 'query' && <QueryInterface />}
        {activeTab === 'validate' && <ValidationTester tools={tools} />}
        {activeTab === 'tools' && <ToolsList tools={tools} />}
        {activeTab === 'metrics' && <MetricsDashboard />}
      </main>

      <footer className="footer">
        <p>L17: Designing Robust Tool Schemas | VAIA Curriculum</p>
      </footer>
    </div>
  );
}

export default App;
