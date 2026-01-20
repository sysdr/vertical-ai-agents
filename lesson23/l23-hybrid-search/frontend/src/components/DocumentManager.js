import React, { useState } from 'react';
import axios from 'axios';

function DocumentManager({ apiUrl, onUpdate }) {
  const [docId, setDocId] = useState('');
  const [docText, setDocText] = useState('');
  const [category, setCategory] = useState('');
  const [year, setYear] = useState('');
  const [author, setAuthor] = useState('');
  const [message, setMessage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!docId || !docText) {
      setMessage({ type: 'error', text: 'ID and text are required' });
      return;
    }

    try {
      const metadata = {};
      if (category) metadata.category = category;
      if (year) metadata.year = parseInt(year);
      if (author) metadata.author = author;

      await axios.post(`${apiUrl}/documents`, {
        id: docId,
        text: docText,
        metadata: metadata
      });

      setMessage({ type: 'success', text: `Document ${docId} added successfully!` });
      
      // Reset form
      setDocId('');
      setDocText('');
      setCategory('');
      setYear('');
      setAuthor('');
      
      // Update stats
      if (onUpdate) onUpdate();

      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to add document' });
    }
  };

  const loadSampleData = async () => {
    const samples = [
      {
        id: 'doc_001',
        text: 'Python FastAPI is a modern web framework for building APIs with automatic documentation and type validation.',
        metadata: { category: 'technology', year: 2024, author: 'Tech Writer' }
      },
      {
        id: 'doc_002',
        text: 'Machine learning models require careful hyperparameter tuning and validation to achieve optimal performance.',
        metadata: { category: 'ai', year: 2024, author: 'ML Engineer' }
      },
      {
        id: 'doc_003',
        text: 'Docker containers provide lightweight, portable environments for application deployment across different platforms.',
        metadata: { category: 'devops', year: 2023, author: 'DevOps Specialist' }
      },
      {
        id: 'doc_004',
        text: 'React hooks revolutionized functional components by enabling state management and lifecycle methods.',
        metadata: { category: 'frontend', year: 2024, author: 'Frontend Dev' }
      },
      {
        id: 'doc_005',
        text: 'Vector databases like ChromaDB enable semantic search by storing and querying high-dimensional embeddings.',
        metadata: { category: 'database', year: 2024, author: 'Database Architect' }
      }
    ];

    try {
      for (const sample of samples) {
        await axios.post(`${apiUrl}/documents`, sample);
      }
      setMessage({ type: 'success', text: `Loaded ${samples.length} sample documents!` });
      if (onUpdate) onUpdate();
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to load sample data' });
    }
  };

  return (
    <div className="document-manager">
      <h2>Add Documents</h2>
      
      {message && (
        <div className={message.type === 'success' ? 'success-message' : 'error-message'}>
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit} className="add-document-form">
        <div className="form-group">
          <label>Document ID *</label>
          <input
            type="text"
            className="form-input"
            placeholder="e.g., doc_123"
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Document Text *</label>
          <textarea
            className="form-textarea"
            placeholder="Enter the document content..."
            value={docText}
            onChange={(e) => setDocText(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>Metadata (Optional)</label>
          <div className="metadata-inputs">
            <input
              type="text"
              className="form-input"
              placeholder="Category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            />
            <input
              type="number"
              className="form-input"
              placeholder="Year"
              value={year}
              onChange={(e) => setYear(e.target.value)}
            />
            <input
              type="text"
              className="form-input"
              placeholder="Author"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
            />
          </div>
        </div>

        <button type="submit" className="submit-button">
          ➕ Add Document
        </button>
      </form>

      <button 
        onClick={loadSampleData}
        className="submit-button"
        style={{ marginTop: '15px', background: '#28a745' }}
      >
        📦 Load Sample Documents
      </button>
    </div>
  );
}

export default DocumentManager;
