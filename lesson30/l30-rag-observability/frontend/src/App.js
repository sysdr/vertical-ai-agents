import React, { useState, useEffect } from 'react';
import MetricsDashboard from './components/MetricsDashboard';
import QueryInterface from './components/QueryInterface';
import './App.css';

function App() {
  const [stats, setStats] = useState(null);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Establish WebSocket connection for real-time metrics
    const websocket = new WebSocket('ws://localhost:8000/ws/metrics');
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStats(data);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWs(websocket);

    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 L30: RAG Observability Dashboard</h1>
        <p>Real-time monitoring of RAG pipeline performance</p>
      </header>
      
      <div className="dashboard-container">
        <QueryInterface />
        <MetricsDashboard stats={stats} />
      </div>
    </div>
  );
}

export default App;
