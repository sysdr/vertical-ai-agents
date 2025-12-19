import React, { useState, useEffect } from 'react';
import AgentExecutor from './components/AgentExecutor';
import DecisionTraceViewer from './components/DecisionTraceViewer';
import ToolsPanel from './components/ToolsPanel';
import MetricsDisplay from './components/MetricsDisplay';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('executor');
  const [systemStatus, setSystemStatus] = useState({ status: 'checking' });
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Check backend health
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => setSystemStatus({ status: 'offline', error: err.message }));

    // Setup WebSocket
    const websocket = new WebSocket('ws://localhost:8000/ws');
    websocket.onopen = () => console.log('WebSocket connected');
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message:', data);
    };
    setWs(websocket);

    return () => {
      if (websocket) websocket.close();
    };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 L8: Agent Decision Maker</h1>
        <div className="status-badge" data-status={systemStatus.status}>
          {systemStatus.status}
        </div>
      </header>

      <nav className="tab-nav">
        <button 
          className={activeTab === 'executor' ? 'active' : ''}
          onClick={() => setActiveTab('executor')}
        >
          Execute Goal
        </button>
        <button 
          className={activeTab === 'traces' ? 'active' : ''}
          onClick={() => setActiveTab('traces')}
        >
          Decision Traces
        </button>
        <button 
          className={activeTab === 'tools' ? 'active' : ''}
          onClick={() => setActiveTab('tools')}
        >
          Available Tools
        </button>
        <button 
          className={activeTab === 'metrics' ? 'active' : ''}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'executor' && <AgentExecutor />}
        {activeTab === 'traces' && <DecisionTraceViewer />}
        {activeTab === 'tools' && <ToolsPanel />}
        {activeTab === 'metrics' && <MetricsDisplay />}
      </main>

      <footer className="app-footer">
        <p>L8: Core Agent Theory - Decision Maker with Planning & Deliberation</p>
      </footer>
    </div>
  );
}

export default App;
