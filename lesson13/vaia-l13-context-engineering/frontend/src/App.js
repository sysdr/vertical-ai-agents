import React, { useState } from 'react';
import TokenCounter from './components/TokenCounter';
import Summarizer from './components/Summarizer';
import ContextOptimizer from './components/ContextOptimizer';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('token-counter');
  const [stats, setStats] = useState({
    totalOptimizations: 0,
    totalTokensSaved: 0,
    averageCompressionRatio: 0,
    totalCostSavings: 0
  });

  const updateStats = (newData) => {
    setStats(prev => ({
      totalOptimizations: prev.totalOptimizations + 1,
      totalTokensSaved: prev.totalTokensSaved + (newData.tokens_saved || 0),
      averageCompressionRatio: newData.compression_ratio || prev.averageCompressionRatio,
      totalCostSavings: prev.totalCostSavings + (newData.money_saved || 0)
    }));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎯 VAIA L13: Context Engineering</h1>
        <p>Production-grade context management and optimization</p>
      </header>

      <Dashboard stats={stats} />

      <div className="tabs">
        <button 
          className={activeTab === 'token-counter' ? 'active' : ''}
          onClick={() => setActiveTab('token-counter')}
        >
          📊 Token Counter
        </button>
        <button 
          className={activeTab === 'summarizer' ? 'active' : ''}
          onClick={() => setActiveTab('summarizer')}
        >
          ✂️ Summarizer
        </button>
        <button 
          className={activeTab === 'optimizer' ? 'active' : ''}
          onClick={() => setActiveTab('optimizer')}
        >
          ⚡ Context Optimizer
        </button>
      </div>

      <div className="content">
        {activeTab === 'token-counter' && <TokenCounter />}
        {activeTab === 'summarizer' && <Summarizer onSummarize={updateStats} />}
        {activeTab === 'optimizer' && <ContextOptimizer onOptimize={updateStats} />}
      </div>

      <footer className="App-footer">
        <p>Building on L12: Few-Shot Prompting | Enabling L14: State Management</p>
        <p>Enterprise Context Management • Token Optimization • Cost Reduction</p>
      </footer>
    </div>
  );
}

export default App;
