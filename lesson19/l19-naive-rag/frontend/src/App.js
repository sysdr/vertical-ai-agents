import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [isQuerying, setIsQuerying] = useState(false);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/documents/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus(`✓ Uploaded: ${response.data.chunks_created} chunks created`);
      fetchStats();
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message || 'Unknown error occurred';
      setUploadStatus(`✗ Error: ${errorMessage}`);
    }
  };

  const handleQuery = async () => {
    if (!query.trim()) return;

    setIsQuerying(true);
    setQueryResult(null);

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        question: query,
        top_k: 3
      });
      setQueryResult(response.data);
    } catch (error) {
      setQueryResult({ error: error.message });
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>🔍 Naïve RAG System</h1>
        <p>Document Chunking + In-Memory Retrieval + LLM Generation</p>
      </header>

      <div className="container">
        {/* Stats Panel */}
        <div className="panel stats-panel">
          <h2>System Statistics</h2>
          {stats ? (
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{stats.knowledge_base.total_chunks}</div>
                <div className="stat-label">Total Chunks</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.knowledge_base.total_queries}</div>
                <div className="stat-label">Queries Processed</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">
                  {stats.knowledge_base.avg_retrieval_time_ms}ms
                </div>
                <div className="stat-label">Avg Retrieval Time</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.knowledge_base.unique_keywords}</div>
                <div className="stat-label">Unique Keywords</div>
              </div>
            </div>
          ) : (
            <div className="loading">Loading stats...</div>
          )}
        </div>

        {/* Upload Panel */}
        <div className="panel upload-panel">
          <h2>📄 Upload Document</h2>
          <div className="upload-area">
            <input
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileUpload}
              id="file-upload"
            />
            <label htmlFor="file-upload" className="upload-button">
              Choose File
            </label>
            {uploadedFile && <div className="file-name">{uploadedFile.name}</div>}
          </div>
          {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
        </div>

        {/* Query Panel */}
        <div className="panel query-panel">
          <h2>💬 Ask a Question</h2>
          <div className="query-input-group">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your question..."
              className="query-input"
              onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
            />
            <button
              onClick={handleQuery}
              disabled={isQuerying || !query.trim()}
              className="query-button"
            >
              {isQuerying ? 'Querying...' : 'Ask'}
            </button>
          </div>

          {queryResult && (
            <div className="query-result">
              {queryResult.error ? (
                <div className="error">Error: {queryResult.error}</div>
              ) : (
                <>
                  <div className="answer-section">
                    <h3>Answer:</h3>
                    <p className="answer-text">{queryResult.answer}</p>
                  </div>

                  <div className="metrics-section">
                    <div className="metric">
                      <span className="metric-label">Retrieval:</span>
                      <span className="metric-value">
                        {queryResult.retrieval_time_ms}ms
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Generation:</span>
                      <span className="metric-value">
                        {queryResult.generation_time_ms}ms
                      </span>
                    </div>
                    <div className="metric">
                      <span className="metric-label">Total:</span>
                      <span className="metric-value">
                        {queryResult.total_time_ms}ms
                      </span>
                    </div>
                  </div>

                  <div className="chunks-section">
                    <h3>Retrieved Chunks ({queryResult.chunks_used.length}):</h3>
                    {queryResult.chunks_used.map((chunk, idx) => (
                      <div key={idx} className="chunk-card">
                        <div className="chunk-header">
                          <span className="chunk-id">{chunk.chunk_id}</span>
                          <span className="chunk-score">Score: {chunk.score}</span>
                        </div>
                        <div className="chunk-text">{chunk.text}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Recent Queries */}
        {stats && stats.recent_queries && stats.recent_queries.length > 0 && (
          <div className="panel recent-panel">
            <h2>Recent Queries</h2>
            <div className="recent-queries">
              {stats.recent_queries.map((q, idx) => (
                <div key={idx} className="recent-query">
                  <div className="recent-question">{q.question}</div>
                  <div className="recent-meta">
                    {q.total_time_ms}ms • {q.chunks_used} chunks
                  </div>
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
