import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ToolsPanel() {
  const [tools, setTools] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTools();
  }, []);

  const fetchTools = async () => {
    try {
      const response = await axios.get('http://localhost:8000/tools');
      setTools(response.data.tools);
    } catch (err) {
      console.error('Failed to fetch tools:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">Loading tools...</div>;

  return (
    <div className="tools-panel">
      <h2>Available Tools</h2>
      <p className="info-text">
        Tools the agent can use during execution
      </p>

      <div className="tools-grid">
        {tools.map((tool, idx) => (
          <div key={idx} className="tool-card">
            <div className="tool-header">
              <h3>{tool.name}</h3>
              <span className="tool-type">{tool.type}</span>
            </div>
            <p className="tool-description">{tool.description}</p>
            <div className="tool-parameters">
              <strong>Parameters:</strong>
              <pre>{JSON.stringify(tool.parameters, null, 2)}</pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ToolsPanel;
