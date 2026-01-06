import React, { useState } from 'react';

function ValidationTester({ tools }) {
  const [selectedTool, setSelectedTool] = useState('');
  const [parameters, setParameters] = useState('{}');
  const [result, setResult] = useState(null);

  const handleValidate = async () => {
    try {
      const params = JSON.parse(parameters);
      const response = await fetch('http://localhost:8000/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: selectedTool,
          parameters: params
        })
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
    }
  };

  const presetTests = {
    get_weather: [
      { label: 'Valid', params: { location: 'Paris', unit: 'celsius' } },
      { label: 'Invalid unit', params: { location: 'Paris', unit: 'kelvin' } },
      { label: 'Invalid chars', params: { location: 'Paris<script>', unit: 'celsius' } }
    ],
    get_time: [
      { label: 'Valid', params: { timezone: 'UTC', format: '24h' } },
      { label: 'Invalid format', params: { timezone: 'UTC', format: 'invalid' } }
    ]
  };

  return (
    <div className="validation-tester">
      <div className="section-header">
        <h2>Schema Validation Tester</h2>
        <p>Test tool parameters against Pydantic schemas</p>
      </div>

      <div className="tester-controls">
        <div className="control-group">
          <label>Select Tool:</label>
          <select 
            value={selectedTool} 
            onChange={(e) => setSelectedTool(e.target.value)}
            className="tool-select"
          >
            <option value="">Choose a tool...</option>
            {tools.map(tool => (
              <option key={tool.name} value={tool.name}>{tool.name}</option>
            ))}
          </select>
        </div>

        {selectedTool && presetTests[selectedTool] && (
          <div className="preset-tests">
            <label>Quick Tests:</label>
            {presetTests[selectedTool].map((test, idx) => (
              <button
                key={idx}
                onClick={() => setParameters(JSON.stringify(test.params, null, 2))}
                className="preset-btn"
              >
                {test.label}
              </button>
            ))}
          </div>
        )}

        <div className="control-group">
          <label>Parameters (JSON):</label>
          <textarea
            value={parameters}
            onChange={(e) => setParameters(e.target.value)}
            className="params-input"
            rows="6"
          />
        </div>

        <button 
          onClick={handleValidate} 
          disabled={!selectedTool}
          className="validate-btn"
        >
          Validate Parameters
        </button>
      </div>

      {result && (
        <div className="validation-result">
          <h3>Validation Result</h3>
          <div className={`result-status ${result.valid ? 'valid' : 'invalid'}`}>
            {result.valid ? '✓ Valid Parameters' : '✗ Validation Failed'}
          </div>
          <pre className="json-display">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default ValidationTester;
