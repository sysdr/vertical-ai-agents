import React, { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './RerankingDashboard.css';

const RerankingDashboard = ({ apiUrl }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState(null);

  const sampleDocuments = [
    { id: '1', text: 'Machine learning models require substantial training data for optimal performance', score: 0.75 },
    { id: '2', text: 'Deep neural networks excel at pattern recognition tasks', score: 0.82 },
    { id: '3', text: 'Natural language processing transforms how computers understand text', score: 0.68 },
    { id: '4', text: 'Computer vision enables machines to interpret visual information', score: 0.71 },
    { id: '5', text: 'Reinforcement learning agents learn through trial and error', score: 0.79 },
    { id: '6', text: 'Transfer learning leverages pre-trained models for new tasks', score: 0.73 },
    { id: '7', text: 'Attention mechanisms revolutionized sequence modeling', score: 0.86 },
    { id: '8', text: 'Generative models can create novel content from learned patterns', score: 0.77 },
    { id: '9', text: 'Ensemble methods combine multiple models for better predictions', score: 0.74 },
    { id: '10', text: 'Hyperparameter tuning optimizes model performance', score: 0.69 }
  ];

  const handleRerank = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${apiUrl}/rerank`, {
        query: query,
        documents: sampleDocuments,
        top_k: 5
      });
      setResults(response.data);
    } catch (error) {
      console.error('Reranking failed:', error);
      alert('Reranking failed. Check console for details.');
    }
    setLoading(false);
  };

  const handleCompare = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${apiUrl}/compare`, {
        query: query,
        documents: sampleDocuments
      });
      setComparison(response.data);
    } catch (error) {
      console.error('Comparison failed:', error);
    }
    setLoading(false);
  };

  const getChartData = () => {
    if (!results) return [];
    return results.reranked_documents.map((doc, idx) => ({
      name: `Doc ${idx + 1}`,
      original: results.original_scores[idx] || 0,
      reranked: doc.rerank_score
    }));
  };

  return (
    <div className="dashboard-container">
      <div className="search-section">
        <h2>🔍 Test Reranking</h2>
        <div className="search-controls">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="search-input"
            onKeyPress={(e) => e.key === 'Enter' && handleRerank()}
          />
          <button 
            onClick={handleRerank} 
            disabled={loading}
            className="btn-primary"
          >
            {loading ? '⏳ Processing...' : '🎯 Rerank'}
          </button>
          <button 
            onClick={handleCompare} 
            disabled={loading}
            className="btn-secondary"
          >
            📊 Compare
          </button>
        </div>
      </div>

      {results && (
        <div className="results-section">
          <div className="results-header">
            <h3>✨ Reranking Results</h3>
            <div className="improvement-badge">
              {results.improvement_pct > 0 ? '📈' : '📉'} 
              {Math.abs(results.improvement_pct)}% {results.improvement_pct > 0 ? 'Improvement' : 'Change'}
            </div>
          </div>

          <div className="info-cards">
            <div className="info-card">
              <span className="info-label">Latency:</span>
              <span className="info-value">{results.latency_ms}ms</span>
            </div>
            <div className="info-card">
              <span className="info-label">Cache:</span>
              <span className="info-value">{results.cache_hit ? '✅ Hit' : '❌ Miss'}</span>
            </div>
            <div className="info-card">
              <span className="info-label">Timestamp:</span>
              <span className="info-value">{new Date(results.timestamp).toLocaleTimeString()}</span>
            </div>
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={getChartData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="original" fill="#82ca9d" name="Original Score" />
                <Bar dataKey="reranked" fill="#8884d8" name="Reranked Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="documents-list">
            <h4>Top Reranked Documents</h4>
            {results.reranked_documents.map((doc) => (
              <div key={doc.id} className="document-card">
                <div className="doc-header">
                  <span className="doc-rank">#{doc.rank}</span>
                  <span className="doc-score">Score: {doc.rerank_score.toFixed(3)}</span>
                </div>
                <p className="doc-text">{doc.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {comparison && (
        <div className="comparison-section">
          <h3>📊 Bi-Encoder vs Cross-Encoder Comparison</h3>
          <p className="comparison-stat">
            Ranking changes in top 5: <strong>{comparison.ranking_changes}</strong> documents
          </p>
          
          <div className="comparison-grid">
            <div className="comparison-column">
              <h4>🔹 Bi-Encoder (L21)</h4>
              {comparison.bi_encoder_top_5.map((doc) => (
                <div key={doc.rank} className="comparison-doc">
                  <span className="comp-rank">#{doc.rank}</span>
                  <span className="comp-score">{doc.score.toFixed(3)}</span>
                  <p>{doc.text}</p>
                </div>
              ))}
            </div>
            
            <div className="comparison-column">
              <h4>🔸 Cross-Encoder (L22)</h4>
              {comparison.cross_encoder_top_5.map((doc) => (
                <div key={doc.rank} className="comparison-doc highlighted">
                  <span className="comp-rank">#{doc.rank}</span>
                  <span className="comp-score">{doc.score.toFixed(3)}</span>
                  <p>{doc.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RerankingDashboard;
