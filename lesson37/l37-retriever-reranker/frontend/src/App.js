import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import MetricsDashboard from './components/MetricsDashboard';
import RetrievalResults from './components/RetrievalResults';
import QueryPlanner from './components/QueryPlanner';

const API_URL = 'http://localhost:8000';

const DEMO_QUERIES = [
  'How does Kubernetes autoscaling work?',
  'What is RAG?',
  'Explain RAG systems',
  'FastAPI async patterns',
  'Production RAG and hybrid search',
  'Kubernetes HPA and Cluster Autoscaler',
  'How do rerankers improve retrieval?',
  'Container orchestration with Docker and Kubernetes',
];

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [useReranking, setUseReranking] = useState(true);
  const [useHybrid, setUseHybrid] = useState(false);
  const [events, setEvents] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    // WebSocket connection for real-time updates
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setEvents(prev => [data, ...prev].slice(0, 20));
    };

    // Fetch metrics periodically
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/metrics`);
        setMetrics(response.data);
      } catch (error) {
        console.error('Metrics fetch error:', error);
      }
    }, 3000);

    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleSearch = async (overrideQuery) => {
    const q = overrideQuery !== undefined && overrideQuery !== null ? overrideQuery : query;
    if (!q.trim()) return;

    if (overrideQuery != null) setQuery(overrideQuery);
    setLoading(true);
    setResults(null);

    try {
      const response = await axios.post(`${API_URL}/retrieve`, {
        query: q,
        top_k: 10,
        use_reranking: useReranking,
        use_hybrid: useHybrid
      });

      setResults(response.data);
    } catch (error) {
      console.error('Retrieval error:', error);
      alert('Retrieval failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>🔍 L37: RetrieverAgent & Reranker Tool</h1>
        <p>Multi-stage retrieval: Plan → Search → Rerank</p>
      </header>

      <div className="container">
        <div className="search-section">
          <div className="search-box">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Enter your query (e.g., 'How does Kubernetes autoscaling work?')"
              disabled={loading}
            />
            <button onClick={handleSearch} disabled={loading}>
              {loading ? 'Retrieving...' : 'Retrieve'}
            </button>
          </div>

          <div className="options">
            <label>
              <input
                type="checkbox"
                checked={useReranking}
                onChange={(e) => setUseReranking(e.target.checked)}
              />
              Enable Reranking
            </label>
            <label>
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
              />
              Enable Hybrid Search (Vector + BM25)
            </label>
          </div>

          <div className="demo-queries">
            <span className="demo-queries-label">Demo queries (click to run):</span>
            <div className="demo-chips">
              {DEMO_QUERIES.map((demoQuery, i) => (
                <button
                  key={i}
                  type="button"
                  className="demo-chip"
                  onClick={() => handleSearch(demoQuery)}
                  disabled={loading}
                  title={demoQuery}
                >
                  {demoQuery}
                </button>
              ))}
            </div>
          </div>

          {results && <QueryPlanner subQueries={results.sub_queries} />}
        </div>

        <div className="results-section">
          {results && <RetrievalResults results={results} />}
        </div>

        <div className="metrics-section">
          {metrics && <MetricsDashboard metrics={metrics} events={events} />}
        </div>
      </div>
    </div>
  );
}

export default App;
