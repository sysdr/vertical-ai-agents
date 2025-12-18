#!/bin/bash

# L7: Basic LLM Prompting & JSON Output - Automated Setup
# Creates a production-grade prompt engineering system with JSON parsing

set -e

PROJECT_NAME="l7-prompt-json-system"
# IMPORTANT: Replace this with your own API key from https://aistudio.google.com/app/apikey
# The key below has been disabled due to being leaked
GEMINI_API_KEY="YOUR_API_KEY_HERE"

echo "🚀 Setting up L7: Basic LLM Prompting & JSON Output..."

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,docker,tests}
cd $PROJECT_NAME

# ============================================================================
# BACKEND SETUP
# ============================================================================

echo "📦 Creating Python backend..."

cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
google-generativeai==0.3.2
python-multipart==0.0.6
aiohttp==3.9.1
python-dotenv==1.0.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.26.0
EOF

cat > backend/.env << EOF
GEMINI_API_KEY=$GEMINI_API_KEY
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF

cat > backend/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
import json
import re
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from enum import Enum

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="L7 Prompt Engineering System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class ParseStrategy(str, Enum):
    DIRECT = "direct"
    REGEX = "regex"
    REPAIR = "repair"
    FAILED = "failed"

class PromptRequest(BaseModel):
    instruction: str = Field(..., description="What you want the LLM to do")
    schema: Dict[str, Any] = Field(..., description="Expected JSON schema")
    temperature: float = Field(0.1, ge=0, le=2)

class ParseResult(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    strategy: ParseStrategy
    error: Optional[str] = None
    parse_time_ms: float
    attempts: int

class StructuredResponse(BaseModel):
    prompt_used: str
    raw_response: str
    parsed_data: Optional[Dict[str, Any]]
    parse_result: ParseResult
    validation_errors: List[str] = []
    timestamp: str

# Example schemas for testing
class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    interests: List[str]
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

class ProductInfo(BaseModel):
    product_name: str
    price: float
    category: str
    in_stock: bool
    rating: Optional[float] = None

# ============================================================================
# PROMPT ENGINE
# ============================================================================

class PromptEngine:
    @staticmethod
    def build_json_prompt(instruction: str, schema: dict, examples: Optional[List[dict]] = None) -> str:
        """Constructs a prompt that enforces JSON output"""
        
        prompt_parts = [
            instruction,
            "",
            "You must respond with ONLY valid JSON matching this exact schema:",
            json.dumps(schema, indent=2),
            "",
            "Critical rules:",
            "1. Return ONLY the JSON object",
            "2. No markdown code blocks (no ```json```)",
            "3. No explanatory text before or after",
            "4. All fields are required unless marked optional",
            "5. Use exact field names from schema",
        ]
        
        if examples:
            prompt_parts.extend([
                "",
                "Example valid responses:",
                *[json.dumps(ex, indent=2) for ex in examples]
            ])
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_repair_prompt(original_text: str, error: str) -> str:
        """Creates a prompt for LLM self-correction"""
        return f"""The previous response could not be parsed as JSON.

Original response:
{original_text}

Error encountered:
{error}

Please provide ONLY a corrected JSON response with no additional text."""

# ============================================================================
# JSON PARSER WITH FALLBACK STRATEGIES
# ============================================================================

class JSONParser:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def parse_with_fallback(self, text: str, max_attempts: int = 3) -> ParseResult:
        """Multi-strategy JSON parsing with fallback"""
        start_time = datetime.now()
        
        # Strategy 1: Direct parse
        result = await self._try_direct_parse(text)
        if result.success:
            result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Strategy 2: Regex extraction
        result = await self._try_regex_extraction(text)
        if result.success:
            result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Strategy 3: LLM self-correction
        for attempt in range(max_attempts):
            result = await self._try_llm_repair(text, result.error)
            if result.success:
                result.attempts = attempt + 1
                result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                return result
            text = result.data if result.data else text
        
        result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result
    
    async def _try_direct_parse(self, text: str) -> ParseResult:
        """Strategy 1: Standard JSON parsing"""
        try:
            data = json.loads(text.strip())
            return ParseResult(
                success=True,
                data=data,
                strategy=ParseStrategy.DIRECT,
                parse_time_ms=0,
                attempts=1
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                strategy=ParseStrategy.DIRECT,
                error=str(e),
                parse_time_ms=0,
                attempts=1
            )
    
    async def _try_regex_extraction(self, text: str) -> ParseResult:
        """Strategy 2: Extract JSON from markdown or embedded text"""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',  # Markdown JSON block
            r'```\s*(\{.*?\})\s*```',       # Generic code block
            r'(\{[^{}]*\{.*?\}[^{}]*\})',   # Nested JSON
            r'(\{.*?\})',                    # Simple JSON
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return ParseResult(
                        success=True,
                        data=data,
                        strategy=ParseStrategy.REGEX,
                        parse_time_ms=0,
                        attempts=1
                    )
                except json.JSONDecodeError:
                    continue
        
        return ParseResult(
            success=False,
            strategy=ParseStrategy.REGEX,
            error="No valid JSON found in text",
            parse_time_ms=0,
            attempts=1
        )
    
    async def _try_llm_repair(self, text: str, error: str) -> ParseResult:
        """Strategy 3: Ask LLM to fix its own output"""
        try:
            repair_prompt = PromptEngine.build_repair_prompt(text, error)
            response = await asyncio.to_thread(
                self.model.generate_content,
                repair_prompt
            )
            
            repaired_text = response.text
            data = json.loads(repaired_text.strip())
            
            return ParseResult(
                success=True,
                data=data,
                strategy=ParseStrategy.REPAIR,
                parse_time_ms=0,
                attempts=1
            )
        except Exception as e:
            return ParseResult(
                success=False,
                data=repaired_text if 'repaired_text' in locals() else None,
                strategy=ParseStrategy.REPAIR,
                error=str(e),
                parse_time_ms=0,
                attempts=1
            )

# ============================================================================
# GEMINI CLIENT WITH STRUCTURED OUTPUT
# ============================================================================

class GeminiStructuredClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.parser = JSONParser()
    
    async def generate_structured(
        self,
        instruction: str,
        schema: dict,
        temperature: float = 0.1
    ) -> StructuredResponse:
        """Generate LLM response and parse to structured JSON"""
        
        # Build prompt
        prompt = PromptEngine.build_json_prompt(instruction, schema)
        
        try:
            # Call Gemini API
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                )
            )
            
            raw_text = response.text
            
            # Parse with fallback strategies
            parse_result = await self.parser.parse_with_fallback(raw_text)
            
            # Validate against schema if parsing succeeded
            validation_errors = []
            if parse_result.success and parse_result.data:
                validation_errors = self._validate_schema(parse_result.data, schema)
            
            return StructuredResponse(
                prompt_used=prompt,
                raw_response=raw_text,
                parsed_data=parse_result.data,
                parse_result=parse_result,
                validation_errors=validation_errors,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return StructuredResponse(
                prompt_used=prompt,
                raw_response=str(e),
                parsed_data=None,
                parse_result=ParseResult(
                    success=False,
                    strategy=ParseStrategy.FAILED,
                    error=str(e),
                    parse_time_ms=0,
                    attempts=1
                ),
                validation_errors=[str(e)],
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_schema(self, data: dict, schema: dict) -> List[str]:
        """Basic schema validation"""
        errors = []
        
        # Check required fields
        required_fields = [k for k, v in schema.items() if not str(v).startswith("Optional")]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types (basic)
        for field, value in data.items():
            if field in schema:
                expected_type = schema[field]
                actual_type = type(value).__name__
                if expected_type != actual_type and not str(expected_type).startswith("Optional"):
                    errors.append(f"Field '{field}' has type {actual_type}, expected {expected_type}")
        
        return errors

# ============================================================================
# API ENDPOINTS
# ============================================================================

# Global client
client = GeminiStructuredClient()

# Metrics storage
metrics = {
    "total_requests": 0,
    "successful_parses": 0,
    "failed_parses": 0,
    "strategy_counts": {
        "direct": 0,
        "regex": 0,
        "repair": 0,
        "failed": 0
    },
    "avg_parse_time_ms": 0,
    "parse_times": []
}

@app.get("/")
async def root():
    return {
        "service": "L7 Prompt Engineering System",
        "status": "operational",
        "features": [
            "Smart prompt construction",
            "Multi-strategy JSON parsing",
            "Schema validation",
            "Real-time metrics"
        ]
    }

@app.post("/generate", response_model=StructuredResponse)
async def generate_structured_output(request: PromptRequest):
    """Generate structured JSON output from LLM"""
    
    result = await client.generate_structured(
        instruction=request.instruction,
        schema=request.schema,
        temperature=request.temperature
    )
    
    # Update metrics
    metrics["total_requests"] += 1
    if result.parse_result.success:
        metrics["successful_parses"] += 1
    else:
        metrics["failed_parses"] += 1
    
    metrics["strategy_counts"][result.parse_result.strategy] += 1
    metrics["parse_times"].append(result.parse_result.parse_time_ms)
    metrics["avg_parse_time_ms"] = sum(metrics["parse_times"]) / len(metrics["parse_times"])
    
    return result

@app.get("/metrics")
async def get_metrics():
    """Get parsing metrics"""
    return {
        **metrics,
        "success_rate": (
            metrics["successful_parses"] / metrics["total_requests"] * 100
            if metrics["total_requests"] > 0 else 0
        ),
        "failure_rate": (
            metrics["failed_parses"] / metrics["total_requests"] * 100
            if metrics["total_requests"] > 0 else 0
        )
    }

@app.post("/test/user-profile")
async def test_user_profile(name: str = "Alice Johnson"):
    """Test endpoint: Generate a user profile"""
    
    schema = {
        "name": "str",
        "age": "int",
        "email": "str",
        "interests": "List[str]"
    }
    
    instruction = f"Generate a realistic user profile for a person named {name}. Include their age, email, and 3-5 interests."
    
    return await client.generate_structured(instruction, schema)

@app.post("/test/product-info")
async def test_product_info(category: str = "electronics"):
    """Test endpoint: Generate product information"""
    
    schema = {
        "product_name": "str",
        "price": "float",
        "category": "str",
        "in_stock": "bool",
        "rating": "Optional[float]"
    }
    
    instruction = f"Generate information for a {category} product. Include realistic details."
    
    return await client.generate_structured(instruction, schema)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# ============================================================================
# FRONTEND SETUP
# ============================================================================

echo "🎨 Creating React frontend..."

cat > frontend/package.json << 'EOF'
{
  "name": "l7-prompt-dashboard",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "devDependencies": {
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>L7 Prompt Engineering Dashboard</title>
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
import PromptTester from './components/PromptTester';
import MetricsDashboard from './components/MetricsDashboard';
import './App.css';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(fetchMetrics, 2000);
    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      setIsConnected(response.ok);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎯 L7: Prompt Engineering & JSON Parsing</h1>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {isConnected ? 'Backend Connected' : 'Backend Disconnected'}
        </div>
      </header>

      <div className="content">
        <div className="left-panel">
          <PromptTester onUpdate={fetchMetrics} />
        </div>
        <div className="right-panel">
          <MetricsDashboard metrics={metrics} />
        </div>
      </div>
    </div>
  );
}

export default App;
EOF

cat > frontend/src/App.css << 'EOF'
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
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
}

.App {
  max-width: 1600px;
  margin: 0 auto;
  padding: 20px;
}

.app-header {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.app-header h1 {
  font-size: 28px;
  color: #2c3e50;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 500;
}

.status-indicator.connected {
  background: #d4edda;
  color: #155724;
}

.status-indicator.disconnected {
  background: #f8d7da;
  color: #721c24;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.connected .status-dot {
  background: #28a745;
}

.disconnected .status-dot {
  background: #dc3545;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.left-panel, .right-panel {
  background: white;
  padding: 30px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

@media (max-width: 1200px) {
  .content {
    grid-template-columns: 1fr;
  }
}

.section-title {
  font-size: 20px;
  color: #2c3e50;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e9ecef;
}

button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

textarea, input {
  width: 100%;
  padding: 12px;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  font-size: 14px;
  font-family: inherit;
  transition: border-color 0.2s;
}

textarea:focus, input:focus {
  outline: none;
  border-color: #667eea;
}

textarea {
  min-height: 120px;
  resize: vertical;
}

label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #495057;
}
EOF

cat > frontend/src/components/PromptTester.js << 'EOF'
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
EOF

cat > frontend/src/components/PromptTester.css << 'EOF'
.prompt-tester {
  max-width: 100%;
}

.form-group {
  margin-bottom: 20px;
}

.schema-input {
  font-family: 'Courier New', monospace;
  min-height: 150px;
}

.button-group {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.button-group button {
  flex: 1;
  min-width: 150px;
}

.error-box {
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
}

.result-container {
  margin-top: 30px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
}

.result-container h3 {
  color: #2c3e50;
  margin-bottom: 20px;
}

.result-section {
  background: white;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 15px;
}

.result-section h4 {
  color: #495057;
  margin-bottom: 10px;
  font-size: 16px;
}

.status-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 600;
  margin-bottom: 10px;
}

.status-badge.success {
  background: #d4edda;
  color: #155724;
}

.status-badge.error {
  background: #f8d7da;
  color: #721c24;
}

.json-display {
  background: #2d2d2d;
  color: #f8f8f2;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 13px;
}

.error-list {
  list-style-position: inside;
  color: #721c24;
}

details {
  margin-top: 15px;
  cursor: pointer;
}

details summary {
  font-weight: 600;
  color: #667eea;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
}

details pre {
  margin-top: 10px;
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12px;
}
EOF

cat > frontend/src/components/MetricsDashboard.js << 'EOF'
import React from 'react';
import './MetricsDashboard.css';

function MetricsDashboard({ metrics }) {
  if (!metrics) {
    return (
      <div className="metrics-dashboard">
        <h2 className="section-title">Live Metrics</h2>
        <p className="loading-message">Loading metrics...</p>
      </div>
    );
  }

  const strategyData = [
    { name: 'Direct Parse', count: metrics.strategy_counts.direct, color: '#28a745' },
    { name: 'Regex Extract', count: metrics.strategy_counts.regex, color: '#ffc107' },
    { name: 'LLM Repair', count: metrics.strategy_counts.repair, color: '#17a2b8' },
    { name: 'Failed', count: metrics.strategy_counts.failed, color: '#dc3545' }
  ];

  const maxCount = Math.max(...strategyData.map(d => d.count), 1);

  return (
    <div className="metrics-dashboard">
      <h2 className="section-title">Live Metrics</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{metrics.total_requests}</div>
          <div className="metric-label">Total Requests</div>
        </div>

        <div className="metric-card success">
          <div className="metric-value">{metrics.successful_parses}</div>
          <div className="metric-label">Successful</div>
        </div>

        <div className="metric-card error">
          <div className="metric-value">{metrics.failed_parses}</div>
          <div className="metric-label">Failed</div>
        </div>

        <div className="metric-card">
          <div className="metric-value">{metrics.success_rate.toFixed(1)}%</div>
          <div className="metric-label">Success Rate</div>
        </div>
      </div>

      <div className="chart-section">
        <h3>Parse Strategy Distribution</h3>
        <div className="bar-chart">
          {strategyData.map((strategy, i) => (
            <div key={i} className="bar-row">
              <div className="bar-label">{strategy.name}</div>
              <div className="bar-container">
                <div
                  className="bar-fill"
                  style={{
                    width: `${(strategy.count / maxCount) * 100}%`,
                    background: strategy.color
                  }}
                >
                  <span className="bar-count">{strategy.count}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="performance-section">
        <h3>Performance</h3>
        <div className="performance-stat">
          <span className="stat-label">Average Parse Time:</span>
          <span className="stat-value">{metrics.avg_parse_time_ms.toFixed(2)} ms</span>
        </div>
        <div className="performance-stat">
          <span className="stat-label">Total Parse Operations:</span>
          <span className="stat-value">{metrics.parse_times.length}</span>
        </div>
      </div>

      <div className="insights-section">
        <h3>Insights</h3>
        <div className="insight-item">
          {metrics.strategy_counts.direct / metrics.total_requests > 0.9 ? (
            <span className="insight-good">✅ Excellent! 90%+ direct parse success</span>
          ) : metrics.strategy_counts.direct / metrics.total_requests > 0.7 ? (
            <span className="insight-warning">⚠️ 70-90% direct parse - room for improvement</span>
          ) : (
            <span className="insight-bad">❌ Low direct parse rate - check prompts</span>
          )}
        </div>
        
        {metrics.strategy_counts.repair > 0 && (
          <div className="insight-item">
            <span className="insight-warning">
              🔧 {metrics.strategy_counts.repair} responses required LLM repair
            </span>
          </div>
        )}
        
        {metrics.avg_parse_time_ms > 100 && (
          <div className="insight-item">
            <span className="insight-warning">
              ⚡ Parse time above 100ms - consider optimization
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricsDashboard;
EOF

cat > frontend/src/components/MetricsDashboard.css << 'EOF'
.metrics-dashboard {
  max-width: 100%;
}

.loading-message {
  text-align: center;
  color: #6c757d;
  font-style: italic;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 15px;
  margin-bottom: 30px;
}

.metric-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-card.success {
  background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.metric-card.error {
  background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
}

.metric-value {
  font-size: 32px;
  font-weight: bold;
  margin-bottom: 5px;
}

.metric-label {
  font-size: 14px;
  opacity: 0.9;
}

.chart-section, .performance-section, .insights-section {
  margin-bottom: 30px;
}

.chart-section h3, .performance-section h3, .insights-section h3 {
  color: #2c3e50;
  margin-bottom: 15px;
  font-size: 18px;
}

.bar-chart {
  background: white;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.bar-row {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.bar-row:last-child {
  margin-bottom: 0;
}

.bar-label {
  width: 120px;
  font-size: 14px;
  font-weight: 500;
  color: #495057;
}

.bar-container {
  flex: 1;
  height: 30px;
  background: #f8f9fa;
  border-radius: 15px;
  position: relative;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 15px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 10px;
  transition: width 0.3s ease;
  min-width: 30px;
}

.bar-count {
  color: white;
  font-weight: 600;
  font-size: 12px;
}

.performance-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.performance-stat {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid #f8f9fa;
}

.performance-stat:last-child {
  border-bottom: none;
}

.stat-label {
  color: #6c757d;
}

.stat-value {
  font-weight: 600;
  color: #2c3e50;
}

.insights-section {
  background: white;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.insight-item {
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 6px;
  font-size: 14px;
}

.insight-item:last-child {
  margin-bottom: 0;
}

.insight-good {
  color: #155724;
  background: #d4edda;
  padding: 8px 12px;
  border-radius: 6px;
  display: inline-block;
}

.insight-warning {
  color: #856404;
  background: #fff3cd;
  padding: 8px 12px;
  border-radius: 6px;
  display: inline-block;
}

.insight-bad {
  color: #721c24;
  background: #f8d7da;
  padding: 8px 12px;
  border-radius: 6px;
  display: inline-block;
}
EOF

# ============================================================================
# TESTS
# ============================================================================

echo "🧪 Creating test suite..."

cat > tests/test_parsing.py << 'EOF'
import pytest
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import JSONParser, PromptEngine, ParseStrategy

@pytest.mark.asyncio
async def test_direct_parse_success():
    parser = JSONParser()
    valid_json = '{"name": "Alice", "age": 30}'
    
    result = await parser._try_direct_parse(valid_json)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.DIRECT
    assert result.data == {"name": "Alice", "age": 30}

@pytest.mark.asyncio
async def test_direct_parse_failure():
    parser = JSONParser()
    invalid_json = '{name: Alice, age: 30}'  # Missing quotes
    
    result = await parser._try_direct_parse(invalid_json)
    
    assert result.success == False
    assert result.strategy == ParseStrategy.DIRECT
    assert result.error is not None

@pytest.mark.asyncio
async def test_regex_extraction_markdown():
    parser = JSONParser()
    text_with_markdown = '''
Here is the JSON you requested:

```json
{"name": "Bob", "age": 25}
```

Hope this helps!
'''
    
    result = await parser._try_regex_extraction(text_with_markdown)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.REGEX
    assert result.data == {"name": "Bob", "age": 25}

@pytest.mark.asyncio
async def test_regex_extraction_embedded():
    parser = JSONParser()
    text_with_embedded = 'The result is {"status": "success", "count": 42} as you can see.'
    
    result = await parser._try_regex_extraction(text_with_embedded)
    
    assert result.success == True
    assert result.strategy == ParseStrategy.REGEX
    assert "status" in result.data

def test_prompt_construction():
    schema = {"name": "str", "age": "int"}
    instruction = "Generate a user"
    
    prompt = PromptEngine.build_json_prompt(instruction, schema)
    
    assert "Generate a user" in prompt
    assert "JSON" in prompt
    assert json.dumps(schema) in prompt
    assert "valid JSON" in prompt.lower()

def test_prompt_construction_with_examples():
    schema = {"status": "str"}
    instruction = "Check status"
    examples = [{"status": "active"}, {"status": "inactive"}]
    
    prompt = PromptEngine.build_json_prompt(instruction, schema, examples)
    
    assert "Example" in prompt
    assert "active" in prompt
    assert "inactive" in prompt

@pytest.mark.asyncio
async def test_parse_with_fallback_success():
    parser = JSONParser()
    valid_json = '{"result": "success"}'
    
    result = await parser.parse_with_fallback(valid_json, max_attempts=1)
    
    assert result.success == True
    assert result.data == {"result": "success"}
    assert result.parse_time_ms >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# ============================================================================
# DOCKER SETUP
# ============================================================================

echo "🐳 Creating Docker configuration..."

cat > docker/Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker/Dockerfile.frontend << 'EOF'
FROM node:18-alpine

WORKDIR /app

COPY frontend/package.json .
RUN npm install

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "start"]
EOF

cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ../backend:/app
    restart: unless-stopped

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    volumes:
      - ../frontend/src:/app/src
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    restart: unless-stopped
EOF

# ============================================================================
# BUILD SCRIPTS
# ============================================================================

echo "📝 Creating build scripts..."

cat > build.sh << 'EOF'
#!/bin/bash
set -e

echo "🏗️  Building L7 Prompt Engineering System..."

# Backend setup
echo "Setting up Python backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Frontend setup
echo "Setting up React frontend..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "  ./start.sh  - Start the system"
echo "  ./test.sh   - Run tests"
EOF

chmod +x build.sh

cat > start.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting L7 Prompt Engineering System..."

# Start backend
echo "Starting Python backend on http://localhost:8000"
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3

# Start frontend
echo "Starting React frontend on http://localhost:3000"
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ System running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
wait
EOF

chmod +x start.sh

cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping L7 Prompt Engineering System..."

# Stop backend
pkill -f "uvicorn main:app"

# Stop frontend
pkill -f "react-scripts start"

echo "✅ System stopped"
EOF

chmod +x stop.sh

cat > test.sh << 'EOF'
#!/bin/bash

echo "🧪 Running L7 tests..."

cd backend
source venv/bin/activate

# Run pytest
python -m pytest ../tests/test_parsing.py -v

deactivate
cd ..

echo "✅ Tests complete!"
EOF

chmod +x test.sh

# ============================================================================
# DOCKER BUILD SCRIPT
# ============================================================================

cat > docker-build.sh << 'EOF'
#!/bin/bash
set -e

echo "🐳 Building Docker containers..."

export GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_API_KEY_HERE}"

cd docker
docker-compose build
echo "✅ Docker build complete!"
echo ""
echo "To start:"
echo "  cd docker && docker-compose up"
EOF

chmod +x docker-build.sh

cat > docker-start.sh << 'EOF'
#!/bin/bash

export GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_API_KEY_HERE}"

echo "🐳 Starting Docker containers..."
cd docker
docker-compose up
EOF

chmod +x docker-start.sh

# ============================================================================
# README
# ============================================================================

cat > README.md << 'EOF'
# L7: Basic LLM Prompting & JSON Output

Production-grade prompt engineering system with multi-strategy JSON parsing and validation.

## Features

- ✅ Smart prompt construction with schema enforcement
- ✅ Multi-strategy JSON parsing (direct, regex, LLM repair)
- ✅ Real-time metrics dashboard
- ✅ Schema validation with Pydantic
- ✅ Error recovery and observability

## Quick Start

### Option 1: Local Development

```bash
# Build
./build.sh

# Start system
./start.sh

# Run tests
./test.sh

# Stop
./stop.sh
```

### Option 2: Docker

```bash
# Build containers
./docker-build.sh

# Start
./docker-start.sh
```

## Access

- Frontend Dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Testing the System

1. Open the dashboard at http://localhost:3000
2. Try the "Test: User Profile" button for quick demo
3. Customize instructions and schemas
4. Watch real-time metrics update

## API Endpoints

- `POST /generate` - Generate structured JSON from prompt
- `GET /metrics` - Get parsing metrics
- `POST /test/user-profile` - Test user profile generation
- `POST /test/product-info` - Test product info generation

## Architecture

- **Backend**: FastAPI with Gemini AI integration
- **Frontend**: React with real-time updates
- **Parsing**: 3-tier fallback strategy
- **Validation**: Pydantic schema enforcement

## Key Concepts

### Parse Strategies

1. **Direct Parse** - Standard json.loads()
2. **Regex Extraction** - Extract from markdown/text
3. **LLM Repair** - Self-correction by the LLM

### Success Metrics

- Direct parse success rate > 90%
- Overall success rate > 95%
- Average parse time < 100ms

## Troubleshooting

**Backend won't start**: Check that port 8000 is free
**Frontend won't start**: Check that port 3000 is free
**CORS errors**: Ensure backend is running first
**Parse failures**: Check prompt structure and schema format

## Next Steps

This lesson prepares for L8 (Core Agent Theory) by providing:
- Structured output parsing for agent perception
- Validation pipelines for decision-making
- Error recovery patterns for production agents
EOF

echo ""
echo "✅ L7 setup complete!"
echo ""
echo "📁 Project created at: ./$PROJECT_NAME"
echo ""
echo "🚀 Quick start:"
echo "   cd $PROJECT_NAME"
echo "   ./build.sh"
echo "   ./start.sh"
echo ""
echo "🌐 Access points:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🧪 Run tests:"
echo "   ./test.sh"
echo ""
echo "🐳 Docker alternative:"
echo "   ./docker-build.sh"
echo "   ./docker-start.sh"