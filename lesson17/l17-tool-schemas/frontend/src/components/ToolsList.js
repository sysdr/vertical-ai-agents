import React, { useState } from 'react';

function ToolsList({ tools }) {
  const [expandedTool, setExpandedTool] = useState(null);

  return (
    <div className="tools-list">
      <div className="section-header">
        <h2>Registered Tool Schemas</h2>
        <p>Browse Pydantic schemas and Gemini function declarations</p>
      </div>

      {tools.map(tool => (
        <div key={tool.name} className="tool-card">
          <div 
            className="tool-header"
            onClick={() => setExpandedTool(expandedTool === tool.name ? null : tool.name)}
          >
            <h3>{tool.name}</h3>
            <span className="expand-icon">
              {expandedTool === tool.name ? '▼' : '▶'}
            </span>
          </div>
          <p className="tool-description">{tool.description}</p>

          {expandedTool === tool.name && (
            <div className="tool-details">
              <div className="schema-section">
                <h4>Input Schema (Pydantic)</h4>
                <pre className="schema-display">
                  {JSON.stringify(tool.input_schema, null, 2)}
                </pre>
              </div>

              <div className="schema-section">
                <h4>Output Schema (Pydantic)</h4>
                <pre className="schema-display">
                  {JSON.stringify(tool.output_schema, null, 2)}
                </pre>
              </div>

              <div className="schema-section">
                <h4>Gemini Declaration</h4>
                <pre className="schema-display">
                  {JSON.stringify(tool.gemini_declaration, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default ToolsList;
