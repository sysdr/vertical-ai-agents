import React, { useState, useEffect } from 'react';
import PromptTester from './components/PromptTester';
import MetricsDashboard from './components/MetricsDashboard';
import './App.css';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      setIsConnected(response.ok);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎯 L7: Prompt Engineering & JSON Parsing</h1>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {isConnected ? 'Backend Connected' : 'Backend Disconnected'}
        </div>
      </header>

      <div className="content">
        <div className="left-panel">
          <PromptTester onUpdate={fetchMetrics} />
        </div>
        <div className="right-panel">
          <MetricsDashboard metrics={metrics} />
        </div>
      </div>
    </div>
  );
}

export default App;
