import React, { useState, useEffect } from 'react';
import ReasoningForm from './components/ReasoningForm';
import ReasoningDisplay from './components/ReasoningDisplay';
import MemoryViewer from './components/MemoryViewer';
import Statistics from './components/Statistics';
import './App.css';

function App() {
  const [currentReasoning, setCurrentReasoning] = useState(null);
  const [memory, setMemory] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchMemory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/memory?limit=10');
      const data = await response.json();
      setMemory(data.traces);
    } catch (error) {
      console.error('Error fetching memory:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  useEffect(() => {
    fetchMemory();
    fetchStats();
    const interval = setInterval(() => {
      fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleReasoningComplete = (result) => {
    setCurrentReasoning(result);
    fetchMemory();
    fetchStats();
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🧠 Chain-of-Thought Reasoning Evaluator</h1>
        <p className="subtitle">L11: Master CoT Prompting & Reasoning Analysis</p>
      </header>

      <div className="container">
        <div className="grid">
          <div className="card">
            <h2>Query Input</h2>
            <ReasoningForm 
              onComplete={handleReasoningComplete}
              loading={loading}
              setLoading={setLoading}
            />
          </div>

          {stats && (
            <div className="card">
              <h2>Quality Statistics</h2>
              <Statistics stats={stats} />
            </div>
          )}
        </div>

        {currentReasoning && (
          <div className="card">
            <h2>Current Reasoning Analysis</h2>
            <ReasoningDisplay reasoning={currentReasoning} />
          </div>
        )}

        <div className="card">
          <h2>Reasoning Memory ({memory.length} traces)</h2>
          <MemoryViewer memory={memory} />
        </div>
      </div>
    </div>
  );
}

export default App;
