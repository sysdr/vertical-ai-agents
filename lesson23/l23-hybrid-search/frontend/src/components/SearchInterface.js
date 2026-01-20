import React, { useState } from 'react';
import axios from 'axios';

function SearchInterface({ apiUrl }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useReranking, setUseReranking] = useState(true);
  const [topK, setTopK] = useState(5);
  
  // Metadata filters
  const [categoryFilter, setCategoryFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');
  const [authorFilter, setAuthorFilter] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      // Build metadata filter
      let metadataFilter = null;
      const conditions = [];
      
      if (categoryFilter) {
        conditions.push({ category: { $eq: categoryFilter } });
      }
      if (yearFilter) {
        const year = parseInt(yearFilter);
        conditions.push({ year: { $gte: year } });
      }
      if (authorFilter) {
        conditions.push({ author: { $eq: authorFilter } });
      }
      
      if (conditions.length > 0) {
        metadataFilter = conditions.length === 1 
          ? conditions[0] 
          : { $and: conditions };
      }

      const response = await axios.post(`${apiUrl}/search`, {
        query: query,
        metadata_filter: metadataFilter,
        top_k: topK,
        use_reranking: useReranking,
        alpha: 0.5
      });

      setResults(response.data);
    } catch (error) {
      console.error('Search failed:', error);
      alert('Search failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="search-interface">
      <div className="search-controls">
        <div className="search-input-group">
          <input
            type="text"
            className="search-input"
            placeholder="Enter your search query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button 
            className="search-button"
            onClick={handleSearch}
            disabled={loading}
          >
            {loading ? '🔄 Searching...' : '🔍 Search'}
          </button>
        </div>

        <div className="filter-controls">
          <div className="filter-group">
            <label>Category</label>
            <input
              type="text"
              className="filter-input"
              placeholder="e.g., technology, science"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>Year (≥)</label>
            <input
              type="number"
              className="filter-input"
              placeholder="e.g., 2024"
              value={yearFilter}
              onChange={(e) => setYearFilter(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>Author</label>
            <input
              type="text"
              className="filter-input"
              placeholder="e.g., John Doe"
              value={authorFilter}
              onChange={(e) => setAuthorFilter(e.target.value)}
            />
          </div>

          <div className="filter-group">
            <label>Top Results</label>
            <input
              type="number"
              className="filter-input"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
          </div>

          <div className="filter-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useReranking}
                onChange={(e) => setUseReranking(e.target.checked)}
              />
              Use Reranking (L22)
            </label>
          </div>
        </div>
      </div>

      <div className="search-results">
        {loading && <div className="loading">Executing hybrid search...</div>}
        
        {!loading && results.length === 0 && query && (
          <div className="no-results">
            <p>No results found. Try adjusting your filters or query.</p>
          </div>
        )}

        {!loading && results.map((result, index) => (
          <div key={result.id} className="result-card">
            <div className="result-header">
              <span className="result-id">#{index + 1}: {result.id}</span>
              <span className="result-score">
                Score: {result.score.toFixed(4)}
              </span>
            </div>
            <div className="result-text">{result.text}</div>
            <div className="result-metadata">
              <span className="metadata-tag">🏷️ {result.rank_source}</span>
              {Object.entries(result.metadata).map(([key, value]) => (
                <span key={key} className="metadata-tag">
                  {key}: {value}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SearchInterface;
