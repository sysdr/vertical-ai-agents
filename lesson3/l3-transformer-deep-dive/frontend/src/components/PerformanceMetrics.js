import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './PerformanceMetrics.css';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState([]);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Failed to fetch config:', error);
    }
  };

  const generateMockMetrics = () => {
    const newMetrics = [];
    for (let i = 0; i < 50; i++) {
      newMetrics.push({
        request: i + 1,
        latency: 30 + Math.random() * 40,
        throughput: 80 + Math.random() * 40,
        memory: 150 + Math.random() * 50
      });
    }
    setMetrics(newMetrics);
  };

  useEffect(() => {
    generateMockMetrics();
  }, []);

  return (
    <div className="metrics-container">
      <div className="metrics-header">
        <h2>Performance Metrics</h2>
        <p>Real-time monitoring of transformer inference performance</p>
      </div>

      {config && (
        <div className="config-display">
          <h3>Current Configuration</h3>
          <div className="config-grid">
            <div className="config-item">
              <span className="config-label">Model Dimension:</span>
              <span className="config-value">{config.d_model}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Attention Heads:</span>
              <span className="config-value">{config.num_heads}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Layers:</span>
              <span className="config-value">{config.num_layers}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Max Sequence:</span>
              <span className="config-value">{config.max_seq_length}</span>
            </div>
          </div>
        </div>
      )}

      <div className="charts-grid">
        <div className="chart-card">
          <h3>Inference Latency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="latency" stroke="#667eea" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Throughput (req/sec)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="throughput" stroke="#764ba2" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Memory Usage (MB)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="request" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="memory" stroke="#f093fb" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="optimization-tips">
        <h3>🚀 Production Optimization Tips</h3>
        <ul>
          <li><strong>Batch Processing:</strong> Group requests by sequence length for optimal GPU utilization</li>
          <li><strong>KV Caching:</strong> Cache key-value pairs for autoregressive generation (5-10x speedup)</li>
          <li><strong>Quantization:</strong> Use INT8/FP16 precision for 2-4x inference speedup</li>
          <li><strong>Attention Sparsity:</strong> Implement sparse attention patterns for long sequences</li>
        </ul>
      </div>
    </div>
  );
};

export default PerformanceMetrics;
