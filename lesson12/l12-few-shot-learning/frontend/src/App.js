import React, { useState, useEffect } from 'react';
import ClassificationPanel from './components/ClassificationPanel';
import ExampleManager from './components/ExampleManager';
import BenchmarkPanel from './components/BenchmarkPanel';
import MetricsDashboard from './components/MetricsDashboard';
import './styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState('classify');
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎯 Few-Shot Learning System</h1>
        <p className="subtitle">L12: Enterprise Classification with Example-Guided Learning</p>
      </header>

      <nav className="tab-nav">
        <button 
          className={activeTab === 'classify' ? 'active' : ''}
          onClick={() => setActiveTab('classify')}
        >
          🔍 Classify
        </button>
        <button 
          className={activeTab === 'examples' ? 'active' : ''}
          onClick={() => setActiveTab('examples')}
        >
          📚 Examples
        </button>
        <button 
          className={activeTab === 'benchmark' ? 'active' : ''}
          onClick={() => setActiveTab('benchmark')}
        >
          📊 Benchmark
        </button>
        <button 
          className={activeTab === 'metrics' ? 'active' : ''}
          onClick={() => setActiveTab('metrics')}
        >
          📈 Metrics
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'classify' && <ClassificationPanel />}
        {activeTab === 'examples' && <ExampleManager />}
        {activeTab === 'benchmark' && <BenchmarkPanel />}
        {activeTab === 'metrics' && <MetricsDashboard metrics={metrics} />}
      </main>

      <footer className="app-footer">
        <div className="footer-stats">
          {metrics && (
            <>
              <span>Total Classifications: {metrics.total_classifications}</span>
              <span>Avg Latency: {metrics.avg_latency_ms?.toFixed(0)}ms</span>
              <span>Avg Confidence: {(metrics.avg_confidence * 100)?.toFixed(1)}%</span>
            </>
          )}
        </div>
      </footer>
    </div>
  );
}

export default App;
