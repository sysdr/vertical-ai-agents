import React, { useState } from 'react';
import './PromptTester.css';

function PromptTester({ onUpdate }) {
  const [instruction, setInstruction] = useState('Generate a user profile for a software engineer');
  const [schema, setSchema] = useState(JSON.stringify({
    name: "str",
    age: "int",
    email: "str",
    interests: "List[str]"
  }, null, 2));
  const [temperature, setTemperature] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          instruction,
          schema: JSON.parse(schema),
          temperature: parseFloat(temperature)
        })
      });

      const data = await response.json();
      setResult(data);
      onUpdate();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testUserProfile = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/test/user-profile?name=John Doe', {
        method: 'POST'
      });
      const data = await response.json();
      setResult(data);
      onUpdate();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testProductInfo = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/test/product-info?category=laptop', {
        method: 'POST'
      });
      const data = await response.json();
      setResult(data);
      onUpdate();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prompt-tester">
      <h2 className="section-title">Prompt Constructor</h2>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Instruction</label>
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="What should the LLM generate?"
          />
        </div>

        <div className="form-group">
          <label>Expected JSON Schema</label>
          <textarea
            value={schema}
            onChange={(e) => setSchema(e.target.value)}
            placeholder='{"field": "type"}'
            className="schema-input"
          />
        </div>

        <div className="form-group">
          <label>Temperature: {temperature}</label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
          />
        </div>

        <div className="button-group">
          <button type="submit" disabled={loading}>
            {loading ? '⏳ Generating...' : '🚀 Generate'}
          </button>
          <button type="button" onClick={testUserProfile} disabled={loading}>
            Test: User Profile
          </button>
          <button type="button" onClick={testProductInfo} disabled={loading}>
            Test: Product Info
          </button>
        </div>
      </form>

      {error && (
        <div className="error-box">
          <strong>❌ Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-container">
          <h3>Results</h3>
          
          <div className="result-section">
            <h4>Parse Status</h4>
            <div className={`status-badge ${result.parse_result.success ? 'success' : 'error'}`}>
              {result.parse_result.success ? '✅ Success' : '❌ Failed'}
            </div>
            <p><strong>Strategy:</strong> {result.parse_result.strategy}</p>
            <p><strong>Parse Time:</strong> {result.parse_result.parse_time_ms.toFixed(2)}ms</p>
            <p><strong>Attempts:</strong> {result.parse_result.attempts}</p>
          </div>

          {result.parsed_data && (
            <div className="result-section">
              <h4>Parsed JSON</h4>
              <pre className="json-display">
                {JSON.stringify(result.parsed_data, null, 2)}
              </pre>
            </div>
          )}

          {result.validation_errors.length > 0 && (
            <div className="result-section">
              <h4>Validation Errors</h4>
              <ul className="error-list">
                {result.validation_errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          <details className="raw-response">
            <summary>Show Raw Response</summary>
            <pre>{result.raw_response}</pre>
          </details>

          <details className="prompt-used">
            <summary>Show Prompt Used</summary>
            <pre>{result.prompt_used}</pre>
          </details>
        </div>
      )}
    </div>
  );
}

export default PromptTester;
