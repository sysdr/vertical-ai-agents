import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import MetricsDashboard from './components/MetricsDashboard';
import AgentExecutor from './components/AgentExecutor';
import PolicyViewer from './components/PolicyViewer';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [metrics, setMetrics] = useState([]);
  const [summary, setSummary] = useState({});
  const [policies, setPolicies] = useState({});

  useEffect(() => {
    loadPolicies();
    loadMetrics(); // Initial load so dashboard shows data immediately
    const interval = setInterval(loadMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const [metricsRes, summaryRes] = await Promise.all([
        axios.get(`${API_BASE}/metrics/recent?limit=10`),
        axios.get(`${API_BASE}/metrics/summary`)
      ]);
      setMetrics(metricsRes.data.metrics || []);
      setSummary(summaryRes.data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  };

  const loadPolicies = async () => {
    try {
      const response = await axios.get(`${API_BASE}/policies`);
      setPolicies(response.data);
    } catch (error) {
      console.error('Failed to load policies:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎯 L34: Planning Loop Controls & Budgeting</h1>
        <p>Budget-Aware ReAct Agent with Iteration & Token Limits</p>
      </header>

      <div className="container">
        <div className="grid">
          <PolicyViewer policies={policies} />
          <AgentExecutor onExecutionComplete={loadMetrics} />
        </div>
        
        <MetricsDashboard 
          metrics={metrics} 
          summary={summary}
        />
      </div>
    </div>
  );
}

export default App;
