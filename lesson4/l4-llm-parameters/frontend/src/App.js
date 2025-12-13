import React, { useState, useEffect } from 'react';
import './App.css';
import ModelComparison from './components/ModelComparison';
import CostCalculator from './components/CostCalculator';
import PerformanceTester from './components/PerformanceTester';
import ContextAnalyzer from './components/ContextAnalyzer';

function App() {
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('comparison');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data.models);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching models:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="App loading">
        <div className="loader"></div>
        <p>Loading LLM Parameter Analysis Platform...</p>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 LLM Parameter Analysis Platform</h1>
          <p className="subtitle">L4: Understanding Model Parameters, Context Windows & Cost Trade-offs</p>
        </div>
        <div className="header-stats">
          <div className="stat-card">
            <span className="stat-label">Models Analyzed</span>
            <span className="stat-value">{Object.keys(models).length}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Max Context</span>
            <span className="stat-value">2M tokens</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Parameter Range</span>
            <span className="stat-value">7B - 405B+</span>
          </div>
        </div>
      </header>

      <nav className="tab-nav">
        <button 
          className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          📊 Model Comparison
        </button>
        <button 
          className={`tab-button ${activeTab === 'cost' ? 'active' : ''}`}
          onClick={() => setActiveTab('cost')}
        >
          💰 Cost Calculator
        </button>
        <button 
          className={`tab-button ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          ⚡ Performance Tester
        </button>
        <button 
          className={`tab-button ${activeTab === 'context' ? 'active' : ''}`}
          onClick={() => setActiveTab('context')}
        >
          📝 Context Analyzer
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'comparison' && <ModelComparison models={models} />}
        {activeTab === 'cost' && <CostCalculator models={models} />}
        {activeTab === 'performance' && <PerformanceTester models={models} />}
        {activeTab === 'context' && <ContextAnalyzer models={models} />}
      </main>

      <footer className="app-footer">
        <p>VAIA Curriculum - Lesson 4 | Building on L3: Transformer Architecture | Preparing for L5: Model Landscape</p>
      </footer>
    </div>
  );
}

export default App;
