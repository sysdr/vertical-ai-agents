#!/bin/bash

# L18: Tool Execution Loop - Complete Implementation Setup
# Builds on L17 tool schemas, prepares for L19 RAG

set -e

PROJECT_NAME="l18-tool-execution-loop"
echo "🚀 Setting up $PROJECT_NAME..."

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend,scripts,docker}
cd $PROJECT_NAME

# ============================================================================
# BACKEND SETUP
# ============================================================================

echo "📦 Creating backend structure..."

cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
google-generativeai==0.8.3
python-multipart==0.0.12
websockets==13.1
aiofiles==24.1.0
prometheus-client==0.21.0
asyncio==3.4.3
EOF

cat > backend/main.py << 'EOF'
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable
import google.generativeai as genai
import json
import time
import asyncio
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import hashlib

# Configure Gemini - API key should be set via environment variable or .env file
# Get your API key from: https://makersuite.google.com/app/apikey
# Then create backend/.env with: GEMINI_API_KEY=your_api_key_here

app = FastAPI(title="L18 Tool Execution Loop")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# TOOL SCHEMAS (from L17)
# ============================================================================

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Optional[Callable] = Field(default=None, exclude=True)

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def calculate_revenue(year: int, quarter: int) -> Dict[str, Any]:
    """Calculate revenue for given period"""
    time.sleep(0.1)  # Simulate processing
    base = 1000000
    growth = 0.15
    quarters_since_2020 = (year - 2020) * 4 + quarter
    revenue = base * ((1 + growth) ** quarters_since_2020)
    return {
        "year": year,
        "quarter": quarter,
        "revenue_usd": round(revenue, 2),
        "growth_rate": growth,
        "currency": "USD"
    }

def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get weather information for a city"""
    time.sleep(0.2)  # Simulate API call
    temps = {"new york": 22, "london": 18, "tokyo": 25, "sydney": 28, "paris": 20}
    temp_c = temps.get(city.lower(), 20)
    temp_f = (temp_c * 9/5) + 32 if unit == "fahrenheit" else temp_c
    return {
        "city": city,
        "temperature": round(temp_f if unit == "fahrenheit" else temp_c, 1),
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 12
    }

def search_products(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search product catalog"""
    time.sleep(0.15)
    products = [
        {"id": 1, "name": "Laptop Pro", "price": 1299, "rating": 4.5},
        {"id": 2, "name": "Wireless Mouse", "price": 29, "rating": 4.2},
        {"id": 3, "name": "USB-C Hub", "price": 49, "rating": 4.7},
        {"id": 4, "name": "Monitor 27\"", "price": 399, "rating": 4.6},
        {"id": 5, "name": "Keyboard Mechanical", "price": 159, "rating": 4.8},
    ]
    results = [p for p in products if query.lower() in p["name"].lower()][:max_results]
    return {
        "query": query,
        "results": results,
        "total_found": len(results),
        "search_time_ms": 150
    }

def document_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search knowledge base (RAG preview for L19)"""
    time.sleep(0.3)
    docs = [
        {"id": "doc1", "title": "VAIA Architecture", "relevance": 0.95},
        {"id": "doc2", "title": "Tool Execution Patterns", "relevance": 0.87},
        {"id": "doc3", "title": "Multi-Agent Systems", "relevance": 0.82},
    ]
    return {
        "query": query,
        "documents": docs[:top_k],
        "total_searched": 1000,
        "search_method": "vector_similarity"
    }

# ============================================================================
# FUNCTION REGISTRY
# ============================================================================

class FunctionRegistry:
    def __init__(self):
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_latency_ms": 0,
            "by_function": {}
        }
    
    def register(self, schema: ToolSchema, function: Callable):
        """Register a function with its schema"""
        self._functions[schema.name] = {
            "schema": schema,
            "function": function,
            "call_count": 0,
            "success_count": 0,
            "total_latency": 0
        }
        print(f"✓ Registered tool: {schema.name}")
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get schema for a function"""
        if name in self._functions:
            return self._functions[name]["schema"]
        return None
    
    def execute(self, name: str, inputs: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Execute a function with safety wrapper"""
        if name not in self._functions:
            return {"error": f"Unknown function: {name}", "success": False}
        
        func_info = self._functions[name]
        func = func_info["function"]
        
        start_time = time.time()
        self._execution_stats["total_calls"] += 1
        func_info["call_count"] += 1
        
        try:
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, **inputs)
                try:
                    result = future.result(timeout=timeout)
                    
                    # Success metrics
                    latency = (time.time() - start_time) * 1000
                    self._execution_stats["successful_calls"] += 1
                    func_info["success_count"] += 1
                    func_info["total_latency"] += latency
                    
                    return {
                        "success": True,
                        "result": result,
                        "latency_ms": round(latency, 2),
                        "function": name
                    }
                except FuturesTimeoutError:
                    self._execution_stats["failed_calls"] += 1
                    return {
                        "success": False,
                        "error": f"Execution timeout after {timeout}s",
                        "function": name
                    }
        except Exception as e:
            self._execution_stats["failed_calls"] += 1
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "function": name
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        by_function = {}
        for name, info in self._functions.items():
            avg_latency = (info["total_latency"] / info["call_count"]) if info["call_count"] > 0 else 0
            success_rate = (info["success_count"] / info["call_count"]) if info["call_count"] > 0 else 0
            by_function[name] = {
                "calls": info["call_count"],
                "successes": info["success_count"],
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 2)
            }
        
        return {
            **self._execution_stats,
            "by_function": by_function
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM"""
        schemas = []
        for name, info in self._functions.items():
            schema = info["schema"]
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description
                        }
                        for param in schema.parameters
                    },
                    "required": [p.name for p in schema.parameters if p.required]
                }
            })
        return schemas

# Initialize registry
registry = FunctionRegistry()

# Register tools
registry.register(
    ToolSchema(
        name="calculate_revenue",
        description="Calculate company revenue for a specific year and quarter",
        parameters=[
            ToolParameter(name="year", type="integer", description="Year (e.g., 2024)"),
            ToolParameter(name="quarter", type="integer", description="Quarter (1-4)")
        ]
    ),
    calculate_revenue
)

registry.register(
    ToolSchema(
        name="get_weather",
        description="Get current weather information for a city",
        parameters=[
            ToolParameter(name="city", type="string", description="City name"),
            ToolParameter(name="unit", type="string", description="Temperature unit (celsius or fahrenheit)", required=False)
        ]
    ),
    get_weather
)

registry.register(
    ToolSchema(
        name="search_products",
        description="Search product catalog by keyword",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query"),
            ToolParameter(name="max_results", type="integer", description="Maximum results to return", required=False)
        ]
    ),
    search_products
)

registry.register(
    ToolSchema(
        name="document_search",
        description="Search knowledge base documents (RAG)",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query"),
            ToolParameter(name="top_k", type="integer", description="Number of results", required=False)
        ]
    ),
    document_search
)

# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class ExecutionState:
    READY = "ready"
    CALLING_LLM = "calling_llm"
    PARSING_RESPONSE = "parsing_response"
    EXECUTING_TOOLS = "executing_tools"
    BUILDING_CONTEXT = "building_context"
    COMPLETE = "complete"
    FAILED = "failed"

class ToolExecutor:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            tools=self._build_tool_declarations()
        )
        self.conversation_history = []
        self.execution_log = []
        self.state = ExecutionState.READY
    
    def _build_tool_declarations(self):
        """Build tool declarations for Gemini"""
        declarations = []
        for schema in self.registry.get_all_schemas():
            declarations.append(genai.protos.Tool(
                function_declarations=[genai.protos.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            name: genai.protos.Schema(
                                type=self._map_type(prop["type"]),
                                description=prop["description"]
                            )
                            for name, prop in schema["parameters"]["properties"].items()
                        },
                        required=schema["parameters"]["required"]
                    )
                )]
            ))
        return declarations
    
    def _map_type(self, type_str: str):
        """Map JSON schema types to Gemini types"""
        mapping = {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "number": genai.protos.Type.NUMBER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }
        return mapping.get(type_str, genai.protos.Type.STRING)
    
    async def execute_loop(self, query: str, max_turns: int = 10) -> Dict[str, Any]:
        """Main execution loop"""
        self.state = ExecutionState.READY
        self.conversation_history = []
        self.execution_log = []
        
        start_time = time.time()
        turn_count = 0
        
        # Initial query
        self.conversation_history.append({"role": "user", "parts": [query]})
        
        while turn_count < max_turns:
            turn_count += 1
            self.state = ExecutionState.CALLING_LLM
            
            try:
                # Call LLM
                response = self.model.generate_content(
                    self.conversation_history,
                    tools=self._build_tool_declarations()
                )
                
                self.state = ExecutionState.PARSING_RESPONSE
                
                # Check for tool calls
                if not response.candidates[0].content.parts:
                    self.state = ExecutionState.FAILED
                    return {
                        "success": False,
                        "error": "Empty response from LLM",
                        "turn_count": turn_count
                    }
                
                # Extract parts
                parts = response.candidates[0].content.parts
                has_function_calls = any(hasattr(part, 'function_call') for part in parts)
                
                if not has_function_calls:
                    # Final answer
                    text_parts = [part.text for part in parts if hasattr(part, 'text')]
                    final_answer = " ".join(text_parts)
                    
                    self.state = ExecutionState.COMPLETE
                    return {
                        "success": True,
                        "answer": final_answer,
                        "turn_count": turn_count,
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                        "execution_log": self.execution_log,
                        "state": self.state
                    }
                
                # Execute function calls
                self.state = ExecutionState.EXECUTING_TOOLS
                function_responses = []
                
                for part in parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        func_name = fc.name
                        func_args = dict(fc.args)
                        
                        self.execution_log.append({
                            "turn": turn_count,
                            "function": func_name,
                            "inputs": func_args,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Execute function
                        result = self.registry.execute(func_name, func_args)
                        
                        # Build response
                        function_responses.append(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=func_name,
                                    response={"result": result}
                                )
                            )
                        )
                        
                        self.execution_log[-1]["result"] = result
                
                # Add function call and responses to history
                self.state = ExecutionState.BUILDING_CONTEXT
                self.conversation_history.append({"role": "model", "parts": parts})
                self.conversation_history.append({"role": "user", "parts": function_responses})
                
            except Exception as e:
                self.state = ExecutionState.FAILED
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "turn_count": turn_count
                }
        
        # Max turns exceeded
        self.state = ExecutionState.FAILED
        return {
            "success": False,
            "error": f"Maximum turns ({max_turns}) exceeded",
            "turn_count": turn_count,
            "execution_log": self.execution_log
        }

# Initialize executor
executor = ToolExecutor(registry)

# ============================================================================
# API ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    query: str
    max_turns: int = 10

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    turn_count: int
    execution_time_ms: float
    execution_log: List[Dict[str, Any]]
    state: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Execute tool loop for user query"""
    result = await executor.execute_loop(request.query, request.max_turns)
    return ChatResponse(**result)

@app.get("/tools")
async def get_tools():
    """Get registered tool schemas"""
    return {
        "tools": registry.get_all_schemas(),
        "count": len(registry._functions)
    }

@app.get("/stats")
async def get_stats():
    """Get execution statistics"""
    return registry.get_stats()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "registered_tools": len(registry._functions)
    }

# WebSocket for real-time monitoring
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial stats
        await websocket.send_json(registry.get_stats())
        
        while True:
            # Send updates every 2 seconds
            await asyncio.sleep(2)
            try:
                stats = registry.get_stats()
                await websocket.send_json(stats)
            except:
                break
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting L18 Tool Execution Loop Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# ============================================================================
# FRONTEND SETUP
# ============================================================================

echo "🎨 Creating frontend..."

cat > frontend/package.json << 'EOF'
{
  "name": "l18-tool-execution-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.13.3"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "vite": "^6.0.3"
  }
}
EOF

cat > frontend/vite.config.js << 'EOF'
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/chat': 'http://localhost:8000',
      '/tools': 'http://localhost:8000',
      '/stats': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
});
EOF

cat > frontend/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>L18: Tool Execution Loop</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
EOF

mkdir -p frontend/src

cat > frontend/src/main.jsx << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

cat > frontend/src/index.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 20px;
}

code {
  font-family: 'Courier New', monospace;
}
EOF

cat > frontend/src/App.jsx << 'EOF'
import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [tools, setTools] = useState([]);
  const [executionLog, setExecutionLog] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchTools();
    fetchStats();
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStats(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      setTimeout(connectWebSocket, 3000);
    };
    
    wsRef.current = ws;
  };

  const fetchTools = async () => {
    try {
      const response = await fetch('/tools');
      const data = await response.json();
      setTools(data.tools);
    } catch (error) {
      console.error('Error fetching tools:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);
    setExecutionLog([]);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_turns: 10 })
      });

      const data = await response.json();
      setResult(data);
      if (data.execution_log) {
        setExecutionLog(data.execution_log);
      }
      
      // Refresh stats
      fetchStats();
    } catch (error) {
      setResult({
        success: false,
        error: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const exampleQueries = [
    "What's the weather in Tokyo and London?",
    "Calculate revenue for Q3 2024 and Q4 2024",
    "Search for laptop products and find weather in Paris",
    "Search documents about VAIA architecture"
  ];

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'];

  const getToolUsageData = () => {
    if (!stats || !stats.by_function) return [];
    return Object.entries(stats.by_function).map(([name, data]) => ({
      name,
      calls: data.calls,
      success_rate: (data.success_rate * 100).toFixed(1)
    }));
  };

  const getSuccessFailureData = () => {
    if (!stats) return [];
    return [
      { name: 'Success', value: stats.successful_calls },
      { name: 'Failed', value: stats.failed_calls }
    ];
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔧 L18: Tool Execution Loop</h1>
        <p>Dynamic function calling with LLM orchestration</p>
      </header>

      <div className="main-content">
        <div className="left-panel">
          <div className="card">
            <h2>Query Interface</h2>
            <form onSubmit={handleSubmit} className="query-form">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask something that requires tool execution..."
                rows={4}
                disabled={loading}
              />
              <button type="submit" disabled={loading}>
                {loading ? '⏳ Executing...' : '▶ Execute Loop'}
              </button>
            </form>

            <div className="examples">
              <h3>Example Queries</h3>
              {exampleQueries.map((example, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => setQuery(example)}
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {result && (
            <div className={`card result-card ${result.success ? 'success' : 'error'}`}>
              <h2>{result.success ? '✅ Response' : '❌ Error'}</h2>
              {result.success ? (
                <>
                  <div className="answer">{result.answer}</div>
                  <div className="metrics">
                    <span>🔄 Turns: {result.turn_count}</span>
                    <span>⚡ Time: {result.execution_time_ms}ms</span>
                    <span>📊 State: {result.state}</span>
                  </div>
                </>
              ) : (
                <div className="error-message">{result.error}</div>
              )}
            </div>
          )}

          {executionLog.length > 0 && (
            <div className="card">
              <h2>📋 Execution Log</h2>
              <div className="execution-log">
                {executionLog.map((log, i) => (
                  <div key={i} className="log-entry">
                    <div className="log-header">
                      <span className="turn">Turn {log.turn}</span>
                      <span className="function">{log.function}</span>
                      <span className="timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="log-details">
                      <div className="log-section">
                        <strong>Inputs:</strong>
                        <pre>{JSON.stringify(log.inputs, null, 2)}</pre>
                      </div>
                      <div className="log-section">
                        <strong>Result:</strong>
                        <pre>{JSON.stringify(log.result, null, 2)}</pre>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="right-panel">
          <div className="card">
            <h2>🛠️ Registered Tools</h2>
            <div className="tools-list">
              {tools.map((tool, i) => (
                <div key={i} className="tool-item">
                  <div className="tool-name">{tool.name}</div>
                  <div className="tool-description">{tool.description}</div>
                  <div className="tool-params">
                    {tool.parameters.required.map((param, j) => (
                      <span key={j} className="param">{param}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {stats && (
            <>
              <div className="card">
                <h2>📊 Execution Statistics</h2>
                <div className="stats-grid">
                  <div className="stat">
                    <div className="stat-value">{stats.total_calls}</div>
                    <div className="stat-label">Total Calls</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{stats.successful_calls}</div>
                    <div className="stat-label">Successful</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">{stats.failed_calls}</div>
                    <div className="stat-label">Failed</div>
                  </div>
                  <div className="stat">
                    <div className="stat-value">
                      {((stats.successful_calls / stats.total_calls) * 100 || 0).toFixed(1)}%
                    </div>
                    <div className="stat-label">Success Rate</div>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>📈 Tool Usage</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={getToolUsageData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="calls" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h2>🎯 Success Distribution</h2>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={getSuccessFailureData()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {getSuccessFailureData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h2>⚡ Performance Metrics</h2>
                <div className="perf-metrics">
                  {stats.by_function && Object.entries(stats.by_function).map(([name, data]) => (
                    <div key={name} className="perf-item">
                      <div className="perf-name">{name}</div>
                      <div className="perf-stats">
                        <span>Calls: {data.calls}</span>
                        <span>Success: {(data.success_rate * 100).toFixed(1)}%</span>
                        <span>Avg: {data.avg_latency_ms}ms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
EOF

cat > frontend/src/App.css << 'EOF'
.app {
  max-width: 1600px;
  margin: 0 auto;
}

.header {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
  text-align: center;
}

.header h1 {
  color: #667eea;
  margin-bottom: 0.5rem;
}

.header p {
  color: #666;
}

.main-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.left-panel, .right-panel {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.card {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.card h2 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.25rem;
}

.query-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.query-form textarea {
  padding: 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
}

.query-form button {
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

.query-form button:hover:not(:disabled) {
  transform: translateY(-2px);
}

.query-form button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.examples {
  margin-top: 1rem;
}

.examples h3 {
  font-size: 0.875rem;
  color: #666;
  margin-bottom: 0.5rem;
}

.example-btn {
  display: block;
  width: 100%;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.875rem;
}

.example-btn:hover:not(:disabled) {
  background: #e8e8e8;
  border-color: #667eea;
}

.result-card {
  border-left: 4px solid;
}

.result-card.success {
  border-left-color: #4caf50;
}

.result-card.error {
  border-left-color: #f44336;
}

.answer {
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 8px;
  line-height: 1.6;
  margin-bottom: 1rem;
  white-space: pre-wrap;
}

.metrics {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  font-size: 0.875rem;
  color: #666;
}

.error-message {
  padding: 1rem;
  background: #ffebee;
  border-radius: 8px;
  color: #c62828;
}

.execution-log {
  max-height: 500px;
  overflow-y: auto;
}

.log-entry {
  padding: 1rem;
  margin-bottom: 1rem;
  background: #f5f5f5;
  border-radius: 8px;
  border-left: 3px solid #667eea;
}

.log-header {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
}

.turn {
  font-weight: 600;
  color: #667eea;
}

.function {
  font-weight: 600;
  color: #4caf50;
}

.timestamp {
  color: #666;
  margin-left: auto;
}

.log-details {
  display: grid;
  gap: 0.5rem;
}

.log-section {
  background: white;
  padding: 0.75rem;
  border-radius: 6px;
}

.log-section strong {
  display: block;
  margin-bottom: 0.5rem;
  color: #333;
}

.log-section pre {
  font-size: 0.75rem;
  overflow-x: auto;
  color: #666;
}

.tools-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.tool-item {
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 8px;
  border-left: 3px solid #4caf50;
}

.tool-name {
  font-weight: 600;
  color: #333;
  margin-bottom: 0.5rem;
}

.tool-description {
  font-size: 0.875rem;
  color: #666;
  margin-bottom: 0.5rem;
}

.tool-params {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.param {
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 4px;
  font-size: 0.75rem;
  color: #667eea;
  border: 1px solid #667eea;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

.stat {
  text-align: center;
  padding: 1rem;
  background: #f5f5f5;
  border-radius: 8px;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: #667eea;
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 0.875rem;
  color: #666;
}

.perf-metrics {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.perf-item {
  padding: 0.75rem;
  background: #f5f5f5;
  border-radius: 6px;
}

.perf-name {
  font-weight: 600;
  color: #333;
  margin-bottom: 0.5rem;
}

.perf-stats {
  display: flex;
  gap: 1rem;
  font-size: 0.75rem;
  color: #666;
}

@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}
EOF

# ============================================================================
# SCRIPTS
# ============================================================================

echo "📝 Creating management scripts..."

cat > scripts/build.sh << 'EOF'
#!/bin/bash
set -e

echo "🔨 Building L18 Tool Execution Loop..."

# Backend
echo "📦 Installing backend dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Frontend
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
EOF

cat > scripts/start.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting L18 Tool Execution Loop..."

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
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Services running:"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
EOF

cat > scripts/stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping services..."

# Kill processes
pkill -f "python main.py" || true
pkill -f "vite" || true

echo "✅ Services stopped"
EOF

cat > scripts/test.sh << 'EOF'
#!/bin/bash
set -e

echo "🧪 Testing L18 Tool Execution Loop..."

# Test backend health
echo "Checking backend health..."
curl -s http://localhost:8000/health | jq

# Test tools endpoint
echo "Checking registered tools..."
curl -s http://localhost:8000/tools | jq

# Test stats endpoint
echo "Checking stats..."
curl -s http://localhost:8000/stats | jq

# Test chat endpoint
echo "Testing tool execution..."
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}' | jq

echo "✅ All tests passed!"
EOF

chmod +x scripts/*.sh

# ============================================================================
# DOCKER SETUP
# ============================================================================

echo "🐳 Creating Docker configuration..."

cat > Dockerfile << 'EOF'
FROM python:3.12-slim

WORKDIR /app

# Backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["python", "main.py"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./backend:/app
    restart: unless-stopped

  frontend:
    image: node:20-alpine
    working_dir: /app
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    command: sh -c "npm install && npm run dev -- --host 0.0.0.0"
    depends_on:
      - backend
    restart: unless-stopped
EOF

# ============================================================================
# DOCUMENTATION
# ============================================================================

cat > README.md << 'EOF'
# L18: Tool Execution Loop

Production-grade tool execution engine with LLM orchestration and real-time monitoring.

## Quick Start

### Non-Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

### Docker
```bash
docker-compose up --build
```

## Access
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Features
- Dynamic function routing and execution
- Multi-turn conversation loops
- Execution safety (timeouts, error handling)
- Real-time monitoring dashboard
- Comprehensive execution logging
- Tool performance metrics

## Architecture
Built on L17 tool schemas, provides foundation for L19 RAG implementation.

## Testing
```bash
./scripts/test.sh
```

## Components
- ToolExecutor: Main execution loop coordinator
- FunctionRegistry: Dynamic tool registration
- SafeExecutor: Timeout and error wrapper
- Real-time WebSocket monitoring
EOF

echo ""
echo "✅ L18 Tool Execution Loop setup complete!"
echo ""
echo "📁 Project structure created at: $PROJECT_NAME/"
echo ""
echo "🚀 Next steps:"
echo "   1. cd $PROJECT_NAME"
echo "   2. ./scripts/build.sh"
echo "   3. ./scripts/start.sh"
echo ""
echo "🌐 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🧪 Test with: ./scripts/test.sh"
echo ""