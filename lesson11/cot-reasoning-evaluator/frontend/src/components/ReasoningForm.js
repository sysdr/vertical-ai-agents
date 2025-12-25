import React, { useState } from 'react';

function ReasoningForm({ onComplete, loading, setLoading }) {
  const [query, setQuery] = useState('');
  const [style, setStyle] = useState('standard');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, style })
      });
      
      const result = await response.json();
      
      if (!response.ok) {
        // Handle error response
        const errorMessage = result.detail || result.message || 'Failed to process reasoning';
        alert(`Error: ${errorMessage}\n\nPlease check:\n1. GEMINI_API_KEY is set correctly\n2. The API key is valid and not leaked\n3. You have sufficient API quota`);
        return;
      }
      
      onComplete(result);
      setQuery('');
    } catch (error) {
      console.error('Error:', error);
      alert(`Failed to process reasoning: ${error.message}\n\nPlease ensure the backend server is running.`);
    } finally {
      setLoading(false);
    }
  };

  const examples = [
    "If Alice has 5 cookies and gives 2 to Bob, then Bob gives 1 to Charlie, how many cookies does each person have?",
    "A train travels 120 km in 2 hours. How long will it take to travel 300 km at the same speed?",
    "If all cats are mammals and all mammals are animals, are all cats animals?"
  ];

  return (
    <form onSubmit={handleSubmit} className="reasoning-form">
      <div className="form-group">
        <label>Query:</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter a problem requiring step-by-step reasoning..."
          rows="4"
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label>Prompt Style:</label>
        <select value={style} onChange={(e) => setStyle(e.target.value)} disabled={loading}>
          <option value="standard">Standard CoT</option>
          <option value="detailed">Detailed CoT</option>
        </select>
      </div>

      <button type="submit" disabled={loading || !query.trim()} className="btn-primary">
        {loading ? 'Reasoning...' : 'Analyze with CoT'}
      </button>

      <div className="examples">
        <p><strong>Example queries:</strong></p>
        {examples.map((ex, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setQuery(ex)}
            className="btn-example"
            disabled={loading}
          >
            {ex.substring(0, 50)}...
          </button>
        ))}
      </div>
    </form>
  );
}

export default ReasoningForm;
