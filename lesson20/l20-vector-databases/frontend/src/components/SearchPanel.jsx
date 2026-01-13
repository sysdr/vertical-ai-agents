import React, { useState } from 'react';

function SearchPanel({ onSearch, results, error }) {
  const [query, setQuery] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (query) onSearch(query);
  };

  return (
    <div className="panel">
      <h2>🔍 Semantic Search</h2>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          placeholder="Enter search query..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>

      <div className="results">
        {error ? (
          <div style={{ padding: '1rem', color: '#d32f2f', textAlign: 'center', background: '#ffebee', borderRadius: '8px', border: '1px solid #ef5350' }}>
            <strong>Error:</strong> {error}
          </div>
        ) : results && results.length > 0 ? (
          results.map((result, idx) => (
            <div key={idx} className="result-card">
              <div className="result-header">
                <span className="similarity">
                  {(result.similarity * 100).toFixed(1)}% match
                </span>
                <span className="source">{result.metadata?.source || 'Unknown'}</span>
              </div>
              <p className="result-text">{result.document}</p>
              <div className="result-meta">
                Chunk {result.metadata?.chunk_index + 1}/{result.metadata?.chunk_total}
              </div>
            </div>
          ))
        ) : (
          <p style={{ padding: '1rem', color: '#666', textAlign: 'center' }}>
            No results found. Try searching with a different query.
          </p>
        )}
      </div>
    </div>
  );
}

export default SearchPanel;
