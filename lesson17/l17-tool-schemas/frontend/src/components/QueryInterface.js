import React, { useState } from 'react';

function QueryInterface() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const examples = [
    "What's the weather in Paris?",
    "Get the current time in New York",
    "Search for information about Pydantic validation"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      const data = await response.json();
      if (!response.ok) {
        // Handle API errors with better messages
        if (data.detail) {
          if (typeof data.detail === 'object' && data.detail.message) {
            setResult({ error: data.detail.message });
          } else {
            setResult({ error: typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail) });
          }
        } else {
          setResult({ error: `Server error: ${response.status}` });
        }
      } else {
        setResult(data);
      }
    } catch (error) {
      setResult({ error: error.message || 'Network error. Please check if the backend is running.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-interface">
      <div className="section-header">
        <h2>Natural Language Query</h2>
        <p>Ask questions that require tool usage - Gemini will automatically call validated tools</p>
      </div>

      <form onSubmit={handleSubmit} className="query-form">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything that requires tool usage..."
          className="query-input"
          rows="3"
        />
        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? 'Processing...' : 'Send Query'}
        </button>
      </form>

      <div className="examples">
        <p className="examples-label">Try these examples:</p>
        {examples.map((ex, idx) => (
          <button
            key={idx}
            onClick={() => setQuery(ex)}
            className="example-btn"
          >
            {ex}
          </button>
        ))}
      </div>

      {result && (
        <div className="result-container">
          <h3>Response</h3>
          {result.error ? (
            <div className="error-message">{result.error}</div>
          ) : (
            <>
              {result.tool_calls && result.tool_calls.length > 0 && (
                <div className="tool-calls">
                  <h4>Tool Calls ({result.tool_calls.length})</h4>
                  {result.tool_calls.map((call, idx) => (
                    <div key={idx} className="tool-call">
                      <div className="tool-call-header">
                        <strong>{call.tool}</strong>
                        <span className={`status ${call.result.success !== false ? 'success' : 'error'}`}>
                          {call.result.success !== false ? '✓ Valid' : '✗ Failed'}
                        </span>
                      </div>
                      <pre className="json-display">
                        {JSON.stringify(call, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              )}
              <div className="final-response">
                <h4>Final Answer</h4>
                <p>{result.response}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default QueryInterface;
