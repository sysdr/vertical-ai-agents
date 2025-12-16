import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Play, RefreshCw, TrendingUp, Zap, DollarSign } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  
  const defaultPrompts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate Fibonacci numbers",
    "What are the key differences between REST and GraphQL?"
  ];

  useEffect(() => {
    loadModels();
    loadRecommendations();
  }, []);

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/models`);
      setModels(response.data.models);
      setSelectedModels(response.data.models.slice(0, 3).map(m => m.id));
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const loadRecommendations = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/recommendations`, {
        params: {
          max_latency_ms: 200,
          max_cost_per_request: 0.01
        }
      });
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error loading recommendations:', error);
    }
  };

  const runBenchmark = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/api/benchmark`, {
        prompts: defaultPrompts,
        models: selectedModels,
        repetitions: 3,
        temperature: 0.7
      });
      setBenchmarkResults(response.data.results);
    } catch (error) {
      console.error('Error running benchmark:', error);
      alert('Benchmark failed: ' + error.message);
    }
    setLoading(false);
  };

  const toggleModel = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const formatLatency = (ms) => {
    return `${Math.round(ms)}ms`;
  };

  const formatCost = (cost) => {
    return `$${(cost * 1000).toFixed(4)}/1K`;
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🤖 VAIA L5: Model Landscape & Selection</h1>
        <p>Compare LLM performance, cost, and quality metrics</p>
      </header>

      <div className="container">
        {/* Model Selection */}
        <div className="card">
          <h2>📊 Available Models</h2>
          <div className="model-grid">
            {models.map(model => (
              <div
                key={model.id}
                className={`model-card ${selectedModels.includes(model.id) ? 'selected' : ''}`}
                onClick={() => toggleModel(model.id)}
              >
                <h3>{model.name}</h3>
                <div className="model-specs">
                  <span className="badge">{model.category}</span>
                  <p className="spec">Context: {(model.context_window / 1000).toFixed(0)}K tokens</p>
                  <p className="spec">Input: {formatCost(model.input_cost_per_1k)}</p>
                  <p className="spec">Output: {formatCost(model.output_cost_per_1k)}</p>
                </div>
                <p className="description">{model.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Benchmark Controls */}
        <div className="card">
          <h2>🚀 Run Benchmark</h2>
          <p className="info">Selected {selectedModels.length} model(s) • Testing {defaultPrompts.length} prompts</p>
          <button
            className="btn-primary"
            onClick={runBenchmark}
            disabled={loading || selectedModels.length === 0}
          >
            {loading ? (
              <>
                <RefreshCw className="icon spin" />
                Running Benchmark...
              </>
            ) : (
              <>
                <Play className="icon" />
                Run Benchmark
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {benchmarkResults && (
          <>
            <div className="card">
              <h2>📈 Performance Summary</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(benchmarkResults.summary).map(([model, data]) => ({
                  model: model.split('-').slice(-1)[0].toUpperCase(),
                  latency: data.avg_latency_ms,
                  cost: data.avg_cost_per_request * 1000
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="latency" fill="#4ade80" name="Latency (ms)" />
                  <Bar yAxisId="right" dataKey="cost" fill="#fb923c" name="Cost ($/1K requests)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h2>💡 Key Insights</h2>
              <div className="insights-grid">
                {Object.entries(benchmarkResults.summary).map(([model, data]) => (
                  <div key={model} className="insight-card">
                    <h3>{model.split('-').pop().toUpperCase()}</h3>
                    <div className="metric">
                      <Zap size={20} />
                      <span>Latency: {formatLatency(data.avg_latency_ms)}</span>
                    </div>
                    <div className="metric">
                      <DollarSign size={20} />
                      <span>Cost: ${(data.avg_cost_per_request).toFixed(6)}/req</span>
                    </div>
                    <div className="metric">
                      <TrendingUp size={20} />
                      <span>Requests: {data.total_requests}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Recommendations */}
        {recommendations && (
          <div className="card">
            <h2>🎯 Smart Recommendations</h2>
            {recommendations.recommended_models.map((rec, idx) => (
              <div key={idx} className="recommendation">
                <h3>{rec.model}</h3>
                <p><strong>Use Case:</strong> {rec.specs.use_case}</p>
                <div className="rec-specs">
                  <span>Latency: {rec.specs.latency_ms}ms</span>
                  <span>Cost: ${rec.specs.cost_per_request}/req</span>
                  <span>Quality: {rec.specs.quality_score}/100</span>
                </div>
              </div>
            ))}
            {recommendations.routing_strategy && (
              <div className="routing-strategy">
                <h4>📍 Suggested Strategy: {recommendations.routing_strategy.strategy}</h4>
                <p>{recommendations.routing_strategy.description}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
