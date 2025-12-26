import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ExampleManager() {
  const [examples, setExamples] = useState({});
  const [newExample, setNewExample] = useState({
    input: '',
    output: '',
    domain: 'customer_support'
  });

  useEffect(() => {
    fetchExamples();
  }, []);

  const fetchExamples = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/examples');
      setExamples(response.data);
    } catch (error) {
      console.error('Failed to fetch examples:', error);
    }
  };

  const handleAddExample = async () => {
    if (!newExample.input.trim() || !newExample.output.trim()) {
      alert('Please fill in both input and output');
      return;
    }

    try {
      await axios.post('http://localhost:8000/api/examples', newExample);
      setNewExample({ input: '', output: '', domain: 'customer_support' });
      fetchExamples();
      alert('Example added successfully!');
    } catch (error) {
      console.error('Failed to add example:', error);
      alert('Failed to add example');
    }
  };

  const totalExamples = Object.values(examples).reduce((sum, arr) => sum + arr.length, 0);

  return (
    <div className="panel example-manager">
      <h2>Example Management</h2>
      <p className="info">Total Examples: {totalExamples}</p>

      <div className="add-example-form">
        <h3>Add New Example</h3>
        <div className="form-group">
          <label>Input</label>
          <textarea
            value={newExample.input}
            onChange={(e) => setNewExample({...newExample, input: e.target.value})}
            placeholder="Example input text..."
            rows="3"
          />
        </div>

        <div className="form-group">
          <label>Output (Classification)</label>
          <input
            type="text"
            value={newExample.output}
            onChange={(e) => setNewExample({...newExample, output: e.target.value})}
            placeholder="Expected classification..."
          />
        </div>

        <div className="form-group">
          <label>Domain</label>
          <select 
            value={newExample.domain}
            onChange={(e) => setNewExample({...newExample, domain: e.target.value})}
          >
            <option value="customer_support">Customer Support</option>
            <option value="general">General</option>
          </select>
        </div>

        <button onClick={handleAddExample} className="btn-primary">
          ➕ Add Example
        </button>
      </div>

      <div className="examples-list">
        <h3>Existing Examples</h3>
        {Object.entries(examples).map(([domain, domainExamples]) => (
          <div key={domain} className="domain-section">
            <h4>{domain} ({domainExamples.length} examples)</h4>
            <div className="examples-grid">
              {domainExamples.map((ex, idx) => (
                <div key={idx} className="example-card">
                  <div className="example-input">
                    <strong>Input:</strong> {ex.input}
                  </div>
                  <div className="example-output">
                    <strong>Output:</strong> {ex.output}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ExampleManager;
