#!/bin/bash

###############################################################################
# L17: Designing Robust Tool Schemas - Complete Setup Script
# Enterprise-Grade Tool Schema Validation System with Pydantic + Gemini AI
###############################################################################

set -e

PROJECT_NAME="l17-tool-schemas"
GEMINI_API_KEY="YOUR_API_KEY_HERE"

echo "=================================================="
echo "L17: Designing Robust Tool Schemas"
echo "Setting up production-grade schema validation..."
echo "=================================================="

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/{components,services},tests}
cd $PROJECT_NAME

###############################################################################
# BACKEND: FastAPI with Pydantic Schemas
###############################################################################

cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.2
google-generativeai==0.8.3
python-dotenv==1.0.1
pytest==8.3.3
httpx==0.27.2
pydantic-settings==2.5.2
EOF

cat > backend/.env << EOF
GEMINI_API_KEY=$GEMINI_API_KEY
EOF

cat > backend/schemas.py << 'EOF'
"""
Tool Schema Definitions using Pydantic v2
Production-grade validation for LLM tool calling
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
from datetime import datetime

class WeatherToolInput(BaseModel):
    """Get current weather for a location"""
    location: str = Field(
        ..., 
        min_length=2, 
        max_length=100,
        description="City name or location (e.g., 'Paris', 'New York')"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit"
    )
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate location contains only safe characters"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Location cannot be empty")
        
        # Allow letters, numbers, spaces, commas, hyphens
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'")
        if not all(c in allowed_chars for c in cleaned):
            raise ValueError(f"Location contains invalid characters: {cleaned}")
        
        return cleaned

class WeatherToolOutput(BaseModel):
    """Structured weather response"""
    location: str
    temperature: float
    unit: str
    condition: str
    humidity: int
    timestamp: str

class TimeToolInput(BaseModel):
    """Get current time for a timezone"""
    timezone: str = Field(
        default="UTC",
        description="Timezone (e.g., 'UTC', 'America/New_York', 'Europe/Paris')"
    )
    format: Literal["12h", "24h"] = Field(
        default="24h",
        description="Time format"
    )
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone format"""
        cleaned = v.strip()
        # Basic validation - in production, use pytz.all_timezones
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/+-")
        if not all(c in allowed_chars for c in cleaned):
            raise ValueError(f"Invalid timezone format: {cleaned}")
        return cleaned

class TimeToolOutput(BaseModel):
    """Structured time response"""
    timezone: str
    current_time: str
    format: str
    utc_offset: str

class SearchToolInput(BaseModel):
    """Search for information"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize search query"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty")
        return cleaned

class SearchToolOutput(BaseModel):
    """Structured search response"""
    query: str
    results_count: int
    results: list[dict]
    timestamp: str

# Tool metadata for registration
TOOL_SCHEMAS = {
    "get_weather": {
        "input": WeatherToolInput,
        "output": WeatherToolOutput,
        "description": "Get current weather information for any location worldwide"
    },
    "get_time": {
        "input": TimeToolInput,
        "output": TimeToolOutput,
        "description": "Get current time for any timezone"
    },
    "search": {
        "input": SearchToolInput,
        "output": SearchToolOutput,
        "description": "Search for information on any topic"
    }
}
EOF

cat > backend/tool_registry.py << 'EOF'
"""
Production Tool Registry with Pydantic Schema Management
Converts Pydantic models to Gemini AI function declarations
"""
from typing import Type, Callable, Any
from pydantic import BaseModel
import inspect
from schemas import TOOL_SCHEMAS

class ToolRegistry:
    """
    Enterprise-grade tool registry managing validated tools
    Mirrors pattern used by OpenAI Assistants API and Anthropic Claude
    """
    
    def __init__(self):
        self.tools: dict[str, dict] = {}
        self._function_map: dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        function: Callable,
        description: str
    ):
        """Register a tool with validation schemas"""
        self.tools[name] = {
            "name": name,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "description": description,
            "gemini_declaration": self._pydantic_to_gemini(name, input_schema, description)
        }
        self._function_map[name] = function
    
    def _pydantic_to_gemini(
        self, 
        name: str, 
        model: Type[BaseModel], 
        description: str
    ) -> dict:
        """
        Convert Pydantic model to Gemini function declaration
        Critical for type-safe LLM-tool interaction
        """
        schema = model.model_json_schema()
        
        # Extract properties and required fields
        properties = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_def = {
                "type": self._json_type_to_gemini(prop_schema.get("type", "string")),
                "description": prop_schema.get("description", "")
            }
            
            # Handle enums (Literal types)
            if "enum" in prop_schema:
                prop_def["enum"] = prop_schema["enum"]
            
            properties[prop_name] = prop_def
        
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": schema.get("required", [])
            }
        }
    
    def _json_type_to_gemini(self, json_type: str) -> str:
        """Map JSON Schema types to Gemini types"""
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object"
        }
        return type_map.get(json_type, "string")
    
    def get_gemini_tools(self) -> list[dict]:
        """Get all tools in Gemini function calling format"""
        return [tool["gemini_declaration"] for tool in self.tools.values()]
    
    def validate_and_execute(self, tool_name: str, parameters: dict) -> dict:
        """
        Validate parameters and execute tool
        Production error handling with detailed feedback
        """
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        input_schema = tool["input_schema"]
        output_schema = tool["output_schema"]
        function = self._function_map[tool_name]
        
        try:
            # Validate input with Pydantic
            validated_input = input_schema(**parameters)
            
            # Execute function
            result = function(validated_input)
            
            # Validate output
            validated_output = output_schema(**result)
            
            return {
                "success": True,
                "tool": tool_name,
                "result": validated_output.model_dump()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "parameters_received": parameters
            }
    
    def get_tool_info(self, tool_name: str) -> dict:
        """Get detailed tool information"""
        if tool_name not in self.tools:
            return {"error": f"Tool not found: {tool_name}"}
        
        tool = self.tools[tool_name]
        return {
            "name": tool_name,
            "description": tool["description"],
            "input_schema": tool["input_schema"].model_json_schema(),
            "output_schema": tool["output_schema"].model_json_schema(),
            "gemini_declaration": tool["gemini_declaration"]
        }

# Global registry instance
registry = ToolRegistry()
EOF

cat > backend/tools.py << 'EOF'
"""
Actual Tool Implementations
These functions are wrapped by Pydantic schemas for type safety
"""
from datetime import datetime
import random
from schemas import (
    WeatherToolInput, WeatherToolOutput,
    TimeToolInput, TimeToolOutput,
    SearchToolInput, SearchToolOutput
)

def get_weather(input_data: WeatherToolInput) -> dict:
    """
    Get weather for location
    In production: call actual weather API (OpenWeatherMap, etc.)
    """
    # Simulated weather data
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Clear"]
    temp_base = 20 if input_data.unit == "celsius" else 68
    
    temperature = temp_base + random.randint(-10, 15)
    
    return {
        "location": input_data.location,
        "temperature": round(temperature, 1),
        "unit": input_data.unit,
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80),
        "timestamp": datetime.now().isoformat()
    }

def get_time(input_data: TimeToolInput) -> dict:
    """
    Get current time for timezone
    In production: use pytz for accurate timezone handling
    """
    now = datetime.now()
    
    if input_data.format == "12h":
        time_str = now.strftime("%I:%M:%S %p")
    else:
        time_str = now.strftime("%H:%M:%S")
    
    return {
        "timezone": input_data.timezone,
        "current_time": time_str,
        "format": input_data.format,
        "utc_offset": "+00:00"  # Simplified
    }

def search(input_data: SearchToolInput) -> dict:
    """
    Search for information
    In production: integrate with search API (Google, Bing, etc.)
    """
    # Simulated search results
    results = []
    for i in range(min(input_data.max_results, 3)):
        results.append({
            "title": f"Result {i+1} for '{input_data.query}'",
            "snippet": f"Information about {input_data.query}...",
            "url": f"https://example.com/result{i+1}"
        })
    
    return {
        "query": input_data.query,
        "results_count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
EOF

cat > backend/main.py << 'EOF'
"""
FastAPI Application with Gemini AI Tool Calling
Production-grade schema validation and error handling
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from tool_registry import registry
from schemas import TOOL_SCHEMAS
import tools
import os
from typing import Optional

app = FastAPI(title="L17 Tool Schema Validator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize tool registry
for tool_name, schema_info in TOOL_SCHEMAS.items():
    function = getattr(tools, tool_name)
    registry.register_tool(
        name=tool_name,
        input_schema=schema_info["input"],
        output_schema=schema_info["output"],
        function=function,
        description=schema_info["description"]
    )

class QueryRequest(BaseModel):
    query: str

class ValidationRequest(BaseModel):
    tool_name: str
    parameters: dict

@app.get("/")
def root():
    return {
        "service": "L17 Tool Schema Validator",
        "status": "running",
        "tools_registered": len(registry.tools)
    }

@app.get("/tools")
def list_tools():
    """List all registered tools with schemas"""
    tools_info = []
    for tool_name in registry.tools:
        tools_info.append(registry.get_tool_info(tool_name))
    return {
        "tools": tools_info,
        "count": len(tools_info)
    }

@app.post("/validate")
def validate_tool_call(request: ValidationRequest):
    """
    Validate tool parameters without execution
    Useful for debugging and schema testing
    """
    if request.tool_name not in registry.tools:
        raise HTTPException(404, f"Tool not found: {request.tool_name}")
    
    tool = registry.tools[request.tool_name]
    input_schema = tool["input_schema"]
    
    try:
        validated = input_schema(**request.parameters)
        return {
            "valid": True,
            "tool": request.tool_name,
            "validated_parameters": validated.model_dump(),
            "schema": input_schema.model_json_schema()
        }
    except ValidationError as e:
        return {
            "valid": False,
            "tool": request.tool_name,
            "errors": e.errors(),
            "schema": input_schema.model_json_schema()
        }

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process natural language query with Gemini AI
    Demonstrates end-to-end tool calling with validation
    """
    try:
        # Get Gemini tools
        gemini_tools = registry.get_gemini_tools()
        
        # Create model with tools
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            tools=gemini_tools
        )
        
        # Start chat
        chat = model.start_chat()
        
        # Send query
        response = chat.send_message(request.query)
        
        # Process tool calls
        tool_results = []
        final_response = None
        
        for part in response.parts:
            if hasattr(part, 'function_call'):
                fc = part.function_call
                
                # Validate and execute
                result = registry.validate_and_execute(
                    fc.name,
                    dict(fc.args)
                )
                
                tool_results.append({
                    "tool": fc.name,
                    "parameters": dict(fc.args),
                    "result": result
                })
                
                # Send result back to Gemini
                response = chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc.name,
                                response=result
                            )
                        )]
                    )
                )
            
            elif hasattr(part, 'text'):
                final_response = part.text
        
        return {
            "query": request.query,
            "tool_calls": tool_results,
            "response": final_response or "No response generated",
            "tools_available": len(gemini_tools)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Query processing failed: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "tools_count": len(registry.tools)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

###############################################################################
# FRONTEND: React Dashboard
###############################################################################

cat > frontend/package.json << 'EOF'
{
  "name": "l17-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "axios": "^1.7.7"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "eslintConfig": {
    "extends": ["react-app"]
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  },
  "devDependencies": {
    "react-scripts": "5.0.1"
  }
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>L17: Tool Schema Validator</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import ToolsList from './components/ToolsList';
import QueryInterface from './components/QueryInterface';
import ValidationTester from './components/ValidationTester';
import './App.css';

function App() {
  const [tools, setTools] = useState([]);
  const [activeTab, setActiveTab] = useState('query');
  const [status, setStatus] = useState({ connected: false, toolsCount: 0 });

  useEffect(() => {
    loadTools();
    checkHealth();
  }, []);

  const loadTools = async () => {
    try {
      const response = await fetch('http://localhost:8000/tools');
      const data = await response.json();
      setTools(data.tools);
      setStatus(prev => ({ ...prev, toolsCount: data.count }));
    } catch (error) {
      console.error('Failed to load tools:', error);
    }
  };

  const checkHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      setStatus(prev => ({ ...prev, connected: data.status === 'healthy' }));
    } catch (error) {
      setStatus(prev => ({ ...prev, connected: false }));
    }
  };

  return (
    <div className="App">
      <header className="header">
        <div className="header-content">
          <h1>🛡️ L17: Tool Schema Validator</h1>
          <p className="subtitle">Production-Grade Pydantic Schema Validation for LLM Tools</p>
          <div className="status-bar">
            <span className={`status-indicator ${status.connected ? 'connected' : 'disconnected'}`}>
              {status.connected ? '✓ Connected' : '✗ Disconnected'}
            </span>
            <span className="tools-count">{status.toolsCount} tools registered</span>
          </div>
        </div>
      </header>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'query' ? 'active' : ''}`}
          onClick={() => setActiveTab('query')}
        >
          Query Interface
        </button>
        <button 
          className={`tab ${activeTab === 'validate' ? 'active' : ''}`}
          onClick={() => setActiveTab('validate')}
        >
          Validation Tester
        </button>
        <button 
          className={`tab ${activeTab === 'tools' ? 'active' : ''}`}
          onClick={() => setActiveTab('tools')}
        >
          Schema Browser
        </button>
      </div>

      <main className="main-content">
        {activeTab === 'query' && <QueryInterface />}
        {activeTab === 'validate' && <ValidationTester tools={tools} />}
        {activeTab === 'tools' && <ToolsList tools={tools} />}
      </main>

      <footer className="footer">
        <p>L17: Designing Robust Tool Schemas | VAIA Curriculum</p>
      </footer>
    </div>
  );
}

export default App;
EOF

cat > frontend/src/components/QueryInterface.js << 'EOF'
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
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
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
EOF

cat > frontend/src/components/ValidationTester.js << 'EOF'
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
EOF

cat > frontend/src/components/ToolsList.js << 'EOF'
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
EOF

cat > frontend/src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.subtitle {
  font-size: 1.1rem;
  opacity: 0.9;
  margin-bottom: 1rem;
}

.status-bar {
  display: flex;
  gap: 1.5rem;
  margin-top: 1rem;
}

.status-indicator {
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
}

.status-indicator.connected {
  background: rgba(72, 187, 120, 0.2);
  color: #2d3748;
}

.status-indicator.disconnected {
  background: rgba(245, 101, 101, 0.2);
  color: #2d3748;
}

.tools-count {
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 20px;
  font-size: 0.9rem;
}

.tabs {
  background: white;
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.tab {
  padding: 0.75rem 1.5rem;
  border: none;
  background: #f7fafc;
  color: #4a5568;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.2s;
}

.tab:hover {
  background: #edf2f7;
}

.tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.main-content {
  flex: 1;
  max-width: 1200px;
  margin: 2rem auto;
  width: 100%;
  padding: 0 1rem;
}

.section-header {
  margin-bottom: 2rem;
}

.section-header h2 {
  font-size: 1.8rem;
  color: #2d3748;
  margin-bottom: 0.5rem;
}

.section-header p {
  color: #718096;
  font-size: 1rem;
}

.query-form {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  margin-bottom: 1.5rem;
}

.query-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 1rem;
}

.query-input:focus {
  outline: none;
  border-color: #667eea;
}

.submit-btn {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.examples {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  margin-bottom: 2rem;
}

.examples-label {
  color: #4a5568;
  font-weight: 500;
  margin-bottom: 0.75rem;
}

.example-btn {
  display: block;
  width: 100%;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s;
}

.example-btn:hover {
  background: #edf2f7;
  border-color: #cbd5e0;
}

.result-container {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
}

.result-container h3 {
  color: #2d3748;
  margin-bottom: 1rem;
}

.tool-calls {
  margin-bottom: 2rem;
}

.tool-calls h4 {
  color: #4a5568;
  margin-bottom: 1rem;
}

.tool-call {
  background: #f7fafc;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  border-left: 4px solid #667eea;
}

.tool-call-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.status {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 500;
}

.status.success {
  background: #c6f6d5;
  color: #22543d;
}

.status.error {
  background: #fed7d7;
  color: #742a2a;
}

.json-display {
  background: #2d3748;
  color: #68d391;
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.85rem;
  line-height: 1.5;
}

.final-response {
  background: #edf2f7;
  padding: 1.5rem;
  border-radius: 8px;
}

.final-response h4 {
  color: #2d3748;
  margin-bottom: 0.75rem;
}

.final-response p {
  color: #4a5568;
  line-height: 1.6;
}

.error-message {
  background: #fed7d7;
  color: #742a2a;
  padding: 1rem;
  border-radius: 6px;
  border-left: 4px solid #fc8181;
}

.tester-controls {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  margin-bottom: 2rem;
}

.control-group {
  margin-bottom: 1.5rem;
}

.control-group label {
  display: block;
  color: #4a5568;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.tool-select {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
}

.tool-select:focus {
  outline: none;
  border-color: #667eea;
}

.preset-tests {
  margin-bottom: 1.5rem;
}

.preset-tests label {
  display: block;
  color: #4a5568;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.preset-btn {
  padding: 0.5rem 1rem;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
  background: #edf2f7;
  border: 1px solid #cbd5e0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.preset-btn:hover {
  background: #e2e8f0;
}

.params-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  resize: vertical;
}

.params-input:focus {
  outline: none;
  border-color: #667eea;
}

.validate-btn {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
}

.validate-btn:hover:not(:disabled) {
  transform: translateY(-2px);
}

.validate-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.validation-result {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
}

.validation-result h3 {
  color: #2d3748;
  margin-bottom: 1rem;
}

.result-status {
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-weight: 600;
}

.result-status.valid {
  background: #c6f6d5;
  color: #22543d;
  border-left: 4px solid #48bb78;
}

.result-status.invalid {
  background: #fed7d7;
  color: #742a2a;
  border-left: 4px solid #fc8181;
}

.tool-card {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  margin-bottom: 1rem;
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  user-select: none;
}

.tool-header h3 {
  color: #2d3748;
  font-size: 1.3rem;
}

.expand-icon {
  color: #667eea;
  font-size: 1.2rem;
}

.tool-description {
  color: #718096;
  margin-top: 0.5rem;
  line-height: 1.5;
}

.tool-details {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 2px solid #e2e8f0;
}

.schema-section {
  margin-bottom: 1.5rem;
}

.schema-section h4 {
  color: #4a5568;
  margin-bottom: 0.75rem;
}

.schema-display {
  background: #2d3748;
  color: #68d391;
  padding: 1rem;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 0.85rem;
  line-height: 1.5;
}

.footer {
  background: #2d3748;
  color: white;
  text-align: center;
  padding: 1.5rem;
  margin-top: 3rem;
}

.footer p {
  opacity: 0.8;
}

@media (max-width: 768px) {
  .header h1 {
    font-size: 1.8rem;
  }
  
  .tabs {
    flex-direction: column;
  }
  
  .tab {
    width: 100%;
  }
}
EOF

###############################################################################
# TESTS
###############################################################################

cat > tests/test_schemas.py << 'EOF'
"""
Test Pydantic schema validation
"""
import pytest
from pydantic import ValidationError
import sys
sys.path.append('../backend')
from schemas import WeatherToolInput, TimeToolInput, SearchToolInput

def test_weather_valid():
    """Test valid weather input"""
    data = WeatherToolInput(location="Paris", unit="celsius")
    assert data.location == "Paris"
    assert data.unit == "celsius"

def test_weather_invalid_unit():
    """Test invalid temperature unit"""
    with pytest.raises(ValidationError):
        WeatherToolInput(location="Paris", unit="kelvin")

def test_weather_invalid_location():
    """Test location with invalid characters"""
    with pytest.raises(ValidationError):
        WeatherToolInput(location="Paris<script>alert()</script>")

def test_time_valid():
    """Test valid time input"""
    data = TimeToolInput(timezone="UTC", format="24h")
    assert data.timezone == "UTC"
    assert data.format == "24h"

def test_search_valid():
    """Test valid search input"""
    data = SearchToolInput(query="Pydantic validation", max_results=5)
    assert data.query == "Pydantic validation"
    assert data.max_results == 5

def test_search_invalid_max_results():
    """Test max_results out of range"""
    with pytest.raises(ValidationError):
        SearchToolInput(query="test", max_results=25)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

###############################################################################
# HELPER SCRIPTS
###############################################################################

cat > build.sh << 'EOF'
#!/bin/bash
set -e

echo "Building L17 Tool Schema Validator..."

# Backend
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
deactivate
cd ..

# Frontend
cd frontend
if [ ! -d "node_modules" ]; then
    npm install --silent
fi
cd ..

echo "✓ Build complete"
EOF

cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting L17 Tool Schema Validator..."

# Start backend
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
cd frontend
BROWSER=none npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "=================================================="
echo "L17 Tool Schema Validator Running!"
echo "=================================================="
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="

# Store PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# Wait
wait
EOF

cat > stop.sh << 'EOF'
#!/bin/bash

echo "Stopping L17 services..."

# Kill backend
if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null || true
    rm .backend.pid
fi

# Kill frontend
if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null || true
    rm .frontend.pid
fi

# Cleanup ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "✓ Services stopped"
EOF

cat > test.sh << 'EOF'
#!/bin/bash
set -e

echo "Running L17 tests..."

cd tests
python -m pytest test_schemas.py -v
cd ..

echo "✓ All tests passed"
EOF

chmod +x *.sh

###############################################################################
# DOCKER SUPPORT
###############################################################################

cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Backend dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Install Node for frontend
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Copy frontend
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

EXPOSE 8000 3000

CMD ["python", "backend/main.py"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  l17-app:
    build: .
    ports:
      - "8000:8000"
      - "3000:3000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./backend:/app/backend
      - ./frontend/src:/app/frontend/src
EOF

###############################################################################
# DOCUMENTATION
###############################################################################

cat > README.md << 'EOF'
# L17: Designing Robust Tool Schemas

Production-grade tool schema validation system with Pydantic and Gemini AI.

## Quick Start

```bash
# Setup and build
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Features

- **Pydantic Schema Validation**: Type-safe tool parameter validation
- **Gemini AI Integration**: Function calling with validated schemas
- **Interactive Dashboard**: Real-time validation testing
- **Production Patterns**: Error handling, schema registry, type safety

## Architecture

- **Backend**: FastAPI + Pydantic v2 + Gemini AI
- **Frontend**: React with validation UI
- **Validation**: Comprehensive input/output validation
- **Error Recovery**: Structured error responses

## Endpoints

- `GET /tools` - List registered tools
- `POST /validate` - Validate tool parameters
- `POST /query` - Process natural language queries
- `GET /health` - Health check

## Usage

1. Open http://localhost:3000
2. Try query: "What's the weather in Paris?"
3. View validation in real-time
4. Test schemas with custom parameters

## Testing

Tests cover:
- Valid parameter validation
- Invalid parameter rejection
- Field validators
- Range constraints
- Type coercion

## Next Steps

L18 builds ToolExecutor class for dynamic tool execution based on these validated schemas.
EOF

###############################################################################
# COMPLETION
###############################################################################

echo ""
echo "=================================================="
echo "✓ L17 Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. ./build.sh"
echo "  3. ./start.sh"
echo ""
echo "Then open: http://localhost:3000"
echo "=================================================="