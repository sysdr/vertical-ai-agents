import React, { useState } from 'react';

function IndexPanel({ onIndex, error, success, loading }) {
  const [text, setText] = useState('');
  const [source, setSource] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text || !source) return;
    await onIndex(text, source);
    setText('');
    setSource('');
  };

  return (
    <div className="panel">
      <h2>📄 Index Documents</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Source (e.g., doc1.pdf)"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          disabled={loading}
        />
        <textarea
          placeholder="Document text to index..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !text || !source}>
          {loading ? 'Indexing...' : 'Index Document'}
        </button>
      </form>
      {error && (
        <div style={{ marginTop: '1rem', padding: '1rem', color: '#d32f2f', background: '#ffebee', borderRadius: '8px', border: '1px solid #ef5350' }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      {success && (
        <div style={{ marginTop: '1rem', padding: '1rem', color: '#2e7d32', background: '#e8f5e9', borderRadius: '8px', border: '1px solid #4caf50' }}>
          <strong>Success:</strong> {success}
        </div>
      )}
    </div>
  );
}

export default IndexPanel;
