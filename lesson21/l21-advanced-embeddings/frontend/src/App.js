import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FileText, Zap, GitBranch, Search, TrendingUp } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [selectedStrategy, setSelectedStrategy] = useState('semantic');
  const [selectedModel, setSelectedModel] = useState('sentence-transformer');
  const [chunks, setChunks] = useState([]);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth();
    loadStrategies();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      setHealth(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const loadStrategies = async () => {
    try {
      const response = await axios.get(`${API_URL}/strategies`);
      setStrategies(response.data.strategies);
      setModels(response.data.models);
    } catch (error) {
      console.error('Failed to load strategies:', error);
    }
  };

  const handleChunk = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/chunk`, {
        text: text,
        strategy: selectedStrategy,
        params: getStrategyParams(selectedStrategy)
      });
      setChunks(response.data.chunks);
    } catch (error) {
      console.error('Chunking failed:', error);
    }
    setLoading(false);
  };

  const handleCompare = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/compare`, {
        text: text,
        strategies: strategies,
        models: models,
        params: {}
      });
      setComparisonResults(response.data.results);
    } catch (error) {
      console.error('Comparison failed:', error);
    }
    setLoading(false);
  };

  const getStrategyParams = (strategy) => {
    const params = {
      'fixed': { chunk_size: 500, overlap: 50 },
      'semantic': { threshold: 0.5, min_chunk_size: 100 },
      'recursive': { max_size: 1000, overlap: 200 },
      'sliding_window': { window_size: 500, stride: 250 },
      'sentence': { sentences_per_chunk: 3 }
    };
    return params[strategy] || {};
  };

  const renderComparisonChart = () => {
    if (!comparisonResults) return null;

    const chartData = Object.entries(comparisonResults).map(([strategy, models]) => {
      const firstModel = Object.values(models)[0];
      return {
        strategy,
        chunks: firstModel.chunks.length,
        coherence: firstModel.metrics.coherence_score || 0,
        avgSize: firstModel.metrics.avg_chunk_size || 0
      };
    });

    return (
      <div className="chart-container">
        <h3>Strategy Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="strategy" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="chunks" fill="#8884d8" name="Chunk Count" />
            <Bar dataKey="coherence" fill="#82ca9d" name="Coherence" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="header">
        <h1><Zap size={32} /> L21: Advanced Embeddings & Chunking</h1>
        <p>Production RAG Engineering Laboratory</p>
        {health && (
          <div className="health-status">
            <span className="status-dot"></span>
            {health.services.chunking_strategies} strategies | {health.services.embeddings} models
          </div>
        )}
      </header>

      <div className="container">
        <div className="input-section">
          <h2><FileText size={24} /> Document Input</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your document here to analyze different chunking strategies..."
            rows={10}
          />
          
          <div className="controls">
            <div className="control-group">
              <label>Chunking Strategy</label>
              <select value={selectedStrategy} onChange={(e) => setSelectedStrategy(e.target.value)}>
                {strategies.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div className="control-group">
              <label>Embedding Model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {models.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>

            <button onClick={handleChunk} disabled={loading || !text} className="btn-primary">
              <GitBranch size={18} /> Apply Strategy
            </button>

            <button onClick={handleCompare} disabled={loading || !text} className="btn-secondary">
              <TrendingUp size={18} /> Compare All
            </button>
          </div>
        </div>

        {chunks.length > 0 && (
          <div className="results-section">
            <h2><Search size={24} /> Chunks ({chunks.length})</h2>
            <div className="chunks-container">
              {chunks.map((chunk, idx) => (
                <div key={idx} className="chunk-card">
                  <div className="chunk-header">
                    <span className="chunk-index">#{idx + 1}</span>
                    <span className="chunk-size">{chunk.text.length} chars</span>
                  </div>
                  <div className="chunk-text">{chunk.text}</div>
                  {chunk.metadata && (
                    <div className="chunk-metadata">
                      {Object.entries(chunk.metadata).map(([key, value]) => (
                        <span key={key} className="metadata-tag">
                          {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {comparisonResults && (
          <div className="comparison-section">
            {renderComparisonChart()}
            
            <div className="comparison-grid">
              {Object.entries(comparisonResults).map(([strategy, models]) => (
                <div key={strategy} className="strategy-card">
                  <h3>{strategy}</h3>
                  {Object.entries(models).map(([model, data]) => (
                    <div key={model} className="model-results">
                      <h4>{model}</h4>
                      <div className="metrics">
                        <div className="metric">
                          <span className="metric-label">Chunks</span>
                          <span className="metric-value">{data.chunks.length}</span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Avg Size</span>
                          <span className="metric-value">
                            {data.metrics.avg_chunk_size ? data.metrics.avg_chunk_size.toFixed(0) : '0'}
                          </span>
                        </div>
                        <div className="metric">
                          <span className="metric-label">Coherence</span>
                          <span className="metric-value">
                            {data.metrics.coherence_score ? (data.metrics.coherence_score * 100).toFixed(1) : '0.0'}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
