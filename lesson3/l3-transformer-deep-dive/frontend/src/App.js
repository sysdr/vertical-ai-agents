import React, { useState } from 'react';
import './App.css';
import TransformerVisualizer from './components/TransformerVisualizer';
import PerformanceMetrics from './components/PerformanceMetrics';

function App() {
  const [activeTab, setActiveTab] = useState('visualizer');

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-content">
          <h1>🧠 Transformer Architecture Deep Dive</h1>
          <p className="subtitle">L3: Interactive Attention Mechanism Exploration</p>
        </div>
        <div className="tab-navigation">
          <button 
            className={activeTab === 'visualizer' ? 'tab-active' : 'tab'}
            onClick={() => setActiveTab('visualizer')}
          >
            Visualizer
          </button>
          <button 
            className={activeTab === 'metrics' ? 'tab-active' : 'tab'}
            onClick={() => setActiveTab('metrics')}
          >
            Performance
          </button>
        </div>
      </header>

      <main className="App-main">
        {activeTab === 'visualizer' && <TransformerVisualizer />}
        {activeTab === 'metrics' && <PerformanceMetrics />}
      </main>

      <footer className="App-footer">
        <p>Advanced Architectures for Vertical AI Agents | Module 1: Foundations</p>
      </footer>
    </div>
  );
}

export default App;
