import React, { useState } from 'react';
import axios from 'axios';

function ClassificationPanel() {
  const [query, setQuery] = useState('');
  const [taskDescription, setTaskDescription] = useState('Classify the following customer support message into one of these categories: REFUND_REQUEST, SHIPPING_INQUIRY, COMPLAINT, ESCALATION, POSITIVE_FEEDBACK, DAMAGE_CLAIM, TECHNICAL_SUPPORT, CANCELLATION');
  const [domain, setDomain] = useState('customer_support');
  const [numExamples, setNumExamples] = useState(3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClassify = async () => {
    if (!query.trim()) {
      alert('Please enter a query to classify');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/classify', {
        query,
        task_description: taskDescription,
        domain,
        num_examples: numExamples
      });
      setResult(response.data);
    } catch (error) {
      console.error('Classification failed:', error);
      alert('Classification failed. See console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel classification-panel">
      <h2>Text Classification</h2>

      <div className="form-group">
        <label>Input Query</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter text to classify..."
          rows="4"
        />
      </div>

      <div className="form-group">
        <label>Task Description</label>
        <textarea
          value={taskDescription}
          onChange={(e) => setTaskDescription(e.target.value)}
          placeholder="Describe the classification task..."
          rows="3"
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Domain</label>
          <select value={domain} onChange={(e) => setDomain(e.target.value)}>
            <option value="customer_support">Customer Support</option>
            <option value="general">General</option>
          </select>
        </div>

        <div className="form-group">
          <label>Number of Examples (Shot Count)</label>
          <input
            type="number"
            value={numExamples}
            onChange={(e) => setNumExamples(parseInt(e.target.value))}
            min="0"
            max="10"
          />
        </div>
      </div>

      <button 
        onClick={handleClassify} 
        disabled={loading}
        className="btn-primary"
      >
        {loading ? '🔄 Classifying...' : '🎯 Classify'}
      </button>

      {result && (
        <div className="result-card">
          <h3>Classification Result</h3>
          <div className="result-item">
            <span className="label">Classification:</span>
            <span className="value classification">{result.classification}</span>
          </div>
          <div className="result-item">
            <span className="label">Confidence:</span>
            <span className="value">{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="result-item">
            <span className="label">Examples Used:</span>
            <span className="value">{result.examples_used} examples</span>
          </div>
          <div className="result-item">
            <span className="label">Token Count:</span>
            <span className="value">{result.token_count} tokens</span>
          </div>
          <div className="result-item">
            <span className="label">Latency:</span>
            <span className="value">{result.latency_ms.toFixed(0)}ms</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default ClassificationPanel;
