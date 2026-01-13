import React, { useState, useEffect } from 'react';
import IndexPanel from './components/IndexPanel';
import SearchPanel from './components/SearchPanel';
import StatsPanel from './components/StatsPanel';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState({ total_documents: 0 });
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError] = useState(null);
  const [indexError, setIndexError] = useState(null);
  const [indexSuccess, setIndexSuccess] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // HTTP polling for stats updates - NO WEBSOCKET CODE
    let statsPollInterval = null;
    let isMounted = true;

    const pollStats = async () => {
      try {
        const response = await fetch(`${API_URL}/stats/default`);
        if (!response.ok) {
          throw new Error('Failed to fetch stats');
        }
        const data = await response.json();
        if (isMounted) {
          setStats(data);
        }
      } catch (error) {
        // Silently handle errors
      }
    };

    // Initial stats fetch
    pollStats();
    
    // Poll stats every 2 seconds
    statsPollInterval = setInterval(() => {
      if (isMounted) {
        pollStats();
      }
    }, 2000);

    return () => {
      isMounted = false;
      if (statsPollInterval) {
        clearInterval(statsPollInterval);
      }
    };
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats/default`);
      if (!response.ok) {
        throw new Error('Failed to fetch stats');
      }
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Stats fetch error:', error);
    }
  };

  const handleIndex = async (text, source) => {
    setIndexError(null);
    setIndexSuccess(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source, collection: 'default' })
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Indexing failed');
      }
      const result = await response.json();
      setIndexSuccess(`Successfully indexed ${result.chunks} chunks from ${result.source}`);
      await fetchStats();
      setTimeout(() => setIndexSuccess(null), 3000);
    } catch (error) {
      console.error('Indexing error:', error);
      setIndexError(error.message || 'Failed to index document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    setSearchError(null);
    setSearchResults([]);
    if (!query.trim()) {
      setSearchError('Please enter a search query');
      return;
    }
    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, collection: 'default', n_results: 5 })
      });
      if (!response.ok) {
        const error = await response.json();
        console.error('Search error:', error);
        const errorMessage = error.detail || 'Search failed. Please try again.';
        setSearchError(errorMessage);
        setSearchResults([]);
        return;
      }
      const data = await response.json();
      setSearchResults(data.results || []);
      if (data.results && data.results.length === 0) {
        setSearchError('No results found. Try a different query or index some documents first.');
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchError('Network error. Please check your connection and try again.');
      setSearchResults([]);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🗄️ L20: Vector Database Dashboard</h1>
        <p>ChromaDB + Gemini Embeddings - Production Vector Storage</p>
      </header>

      <div className="container">
        <StatsPanel stats={stats} />
        <IndexPanel onIndex={handleIndex} error={indexError} success={indexSuccess} loading={loading} />
        <SearchPanel onSearch={handleSearch} results={searchResults} error={searchError} />
      </div>
    </div>
  );
}

export default App;
