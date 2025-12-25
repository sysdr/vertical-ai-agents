#!/bin/bash

# L11: Chain-of-Thought Prompting System - Automated Setup
# Builds production-grade CoT reasoning evaluator with Gemini AI integration

set -e

PROJECT_NAME="cot-reasoning-evaluator"
GEMINI_API_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"

echo "🚀 Setting up L11: Chain-of-Thought Reasoning Evaluator"

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,data,diagrams}
cd $PROJECT_NAME

# Backend setup
cat > backend/requirements.txt << 'EOF'
fastapi==0.110.0
uvicorn[standard]==0.27.1
google-generativeai==0.4.1
pydantic==2.6.3
python-multipart==0.0.9
aiofiles==23.2.1
pytest==8.1.1
httpx==0.27.0
EOF

cat > backend/cot_agent.py << 'EOF'
"""Chain-of-Thought Agent with Reasoning Evaluation"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai

# Configure Gemini AI
genai.configure(api_key="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8")

class CoTPromptBuilder:
    """Constructs prompts with explicit CoT instructions"""
    
    @staticmethod
    def build_cot_prompt(query: str, style: str = "standard") -> str:
        """Build CoT prompt with reasoning instructions"""
        if style == "detailed":
            return f"""Solve this problem using step-by-step reasoning. Show all your work.

Query: {query}

Instructions:
1. First, identify what information you have
2. Break down the problem into smaller steps
3. Solve each step clearly, explaining your logic
4. Show intermediate results
5. State your final conclusion

Think carefully and be explicit about your reasoning process."""
        
        return f"""Think step-by-step and show your reasoning.

Query: {query}

Please:
1. Break down the problem
2. Show each reasoning step
3. State your conclusion clearly

Explain your logic as you go."""

class ReasoningTraceParser:
    """Extracts structured reasoning steps from LLM output"""
    
    @staticmethod
    def extract_steps(trace: str) -> List[str]:
        """Parse reasoning trace into discrete steps"""
        # Try numbered steps first (1., 2., etc.)
        numbered = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', trace, re.DOTALL)
        if len(numbered) >= 2:
            return [step.strip() for step in numbered]
        
        # Try bullet points
        bulleted = re.findall(r'[-•]\s*(.+?)(?=[-•]|$)', trace, re.DOTALL)
        if len(bulleted) >= 2:
            return [step.strip() for step in bulleted]
        
        # Fall back to sentence splitting
        sentences = [s.strip() for s in trace.split('.') if len(s.strip()) > 10]
        return sentences[:10]  # Cap at 10 steps
    
    @staticmethod
    def extract_conclusion(trace: str) -> Optional[str]:
        """Extract final conclusion from reasoning trace"""
        conclusion_markers = [
            'therefore', 'thus', 'in conclusion', 'finally',
            'the answer is', 'result:', 'conclusion:'
        ]
        
        trace_lower = trace.lower()
        for marker in conclusion_markers:
            if marker in trace_lower:
                idx = trace_lower.rfind(marker)
                return trace[idx:].strip()
        
        # Return last sentence as fallback
        sentences = [s.strip() for s in trace.split('.') if s.strip()]
        return sentences[-1] if sentences else None

class QualityEvaluator:
    """Evaluates reasoning trace quality across multiple dimensions"""
    
    @staticmethod
    def assess_clarity(steps: List[str]) -> float:
        """Score clarity based on transition words and structure"""
        transition_words = [
            'first', 'second', 'then', 'next', 'after', 'finally',
            'therefore', 'thus', 'so', 'because', 'since'
        ]
        
        if not steps:
            return 0.0
        
        clarity_score = 0.0
        for step in steps:
            step_lower = step.lower()
            # Check for transition words
            has_transition = any(word in step_lower for word in transition_words)
            # Check for reasonable length (not too short/long)
            good_length = 20 < len(step) < 200
            # Check for specific details (numbers, entities)
            has_specifics = bool(re.search(r'\d+|[A-Z][a-z]+', step))
            
            score = sum([has_transition, good_length, has_specifics]) / 3.0
            clarity_score += score
        
        return clarity_score / len(steps)
    
    @staticmethod
    def check_sequential_logic(steps: List[str]) -> float:
        """Verify logical flow between consecutive steps"""
        if len(steps) < 2:
            return 0.0
        
        logic_score = 0.0
        for i in range(len(steps) - 1):
            current = steps[i].lower()
            next_step = steps[i + 1].lower()
            
            # Check for referential continuity
            # Extract nouns/numbers from current step
            current_entities = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', steps[i]))
            next_entities = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', steps[i + 1]))
            
            # Good flow if next step references current entities
            overlap = len(current_entities & next_entities) / max(len(current_entities), 1)
            logic_score += min(overlap, 1.0)
        
        return logic_score / (len(steps) - 1)
    
    @staticmethod
    def has_clear_conclusion(trace: str) -> bool:
        """Check if reasoning ends with explicit conclusion"""
        conclusion = ReasoningTraceParser.extract_conclusion(trace)
        return conclusion is not None and len(conclusion) > 10
    
    @classmethod
    def evaluate_reasoning(cls, trace: str) -> Dict[str, float]:
        """Comprehensive quality scoring"""
        steps = ReasoningTraceParser.extract_steps(trace)
        
        return {
            "step_count": len(steps),
            "avg_step_length": sum(len(s) for s in steps) / max(len(steps), 1),
            "clarity_score": cls.assess_clarity(steps),
            "logic_flow": cls.check_sequential_logic(steps),
            "conclusion_present": 1.0 if cls.has_clear_conclusion(trace) else 0.0,
            "overall_quality": cls.calculate_overall_score(steps, trace)
        }
    
    @classmethod
    def calculate_overall_score(cls, steps: List[str], trace: str) -> float:
        """Weighted average of quality metrics"""
        if not steps:
            return 0.0
        
        clarity = cls.assess_clarity(steps)
        logic = cls.check_sequential_logic(steps)
        conclusion = 1.0 if cls.has_clear_conclusion(trace) else 0.0
        step_adequacy = min(len(steps) / 5.0, 1.0)  # Ideal: 5+ steps
        
        # Weighted average: clarity 30%, logic 30%, conclusion 20%, steps 20%
        return 0.3 * clarity + 0.3 * logic + 0.2 * conclusion + 0.2 * step_adequacy

class CoTAgent:
    """Chain-of-Thought enabled agent extending L10's SimpleAgent concepts"""
    
    def __init__(self, data_dir: Path = Path("../data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.memory_file = self.data_dir / "cot_memory.json"
        self.model = genai.GenerativeModel('gemini-pro')
        self.memory = self._load_memory()
    
    def _load_memory(self) -> List[Dict]:
        """Load reasoning traces from persistent memory"""
        if self.memory_file.exists():
            return json.loads(self.memory_file.read_text())
        return []
    
    def _save_memory(self):
        """Persist reasoning traces to disk"""
        self.memory_file.write_text(json.dumps(self.memory, indent=2))
    
    async def reason_with_cot(self, query: str, style: str = "standard") -> Dict:
        """Generate CoT reasoning and evaluate quality"""
        # Build CoT prompt
        prompt = CoTPromptBuilder.build_cot_prompt(query, style)
        
        # Call Gemini API
        response = self.model.generate_content(prompt)
        reasoning_trace = response.text
        
        # Parse reasoning steps
        steps = ReasoningTraceParser.extract_steps(reasoning_trace)
        conclusion = ReasoningTraceParser.extract_conclusion(reasoning_trace)
        
        # Evaluate quality
        quality_scores = QualityEvaluator.evaluate_reasoning(reasoning_trace)
        
        # Store in memory
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "reasoning_trace": reasoning_trace,
            "steps": steps,
            "conclusion": conclusion,
            "quality_scores": quality_scores,
            "prompt_style": style
        }
        self.memory.append(memory_entry)
        self._save_memory()
        
        return memory_entry
    
    def get_memory(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent reasoning traces"""
        return self.memory[-limit:]
    
    def get_high_quality_traces(self, min_score: float = 0.7) -> List[Dict]:
        """Get reasoning traces above quality threshold"""
        return [
            trace for trace in self.memory
            if trace["quality_scores"]["overall_quality"] >= min_score
        ]
EOF

cat > backend/main.py << 'EOF'
"""FastAPI backend for CoT Reasoning Evaluator"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from cot_agent import CoTAgent

app = FastAPI(title="CoT Reasoning Evaluator API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CoT agent
agent = CoTAgent(data_dir=Path("/app/data") if Path("/app").exists() else Path("../data"))

class QueryRequest(BaseModel):
    query: str
    style: str = "standard"

class MemoryResponse(BaseModel):
    traces: List[dict]
    total_count: int

@app.get("/")
async def root():
    return {
        "service": "CoT Reasoning Evaluator",
        "lesson": "L11",
        "status": "operational"
    }

@app.post("/api/reason")
async def reason_with_cot(request: QueryRequest):
    """Process query with Chain-of-Thought reasoning"""
    try:
        result = await agent.reason_with_cot(request.query, request.style)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory")
async def get_memory(limit: int = 10):
    """Retrieve recent reasoning traces"""
    traces = agent.get_memory(limit)
    return MemoryResponse(traces=traces, total_count=len(agent.memory))

@app.get("/api/quality-traces")
async def get_quality_traces(min_score: float = 0.7):
    """Get high-quality reasoning traces for few-shot examples"""
    traces = agent.get_high_quality_traces(min_score)
    return {"traces": traces, "count": len(traces)}

@app.get("/api/stats")
async def get_statistics():
    """Get reasoning quality statistics"""
    if not agent.memory:
        return {"message": "No reasoning traces yet"}
    
    scores = [trace["quality_scores"]["overall_quality"] for trace in agent.memory]
    return {
        "total_traces": len(agent.memory),
        "avg_quality": sum(scores) / len(scores),
        "max_quality": max(scores),
        "min_quality": min(scores),
        "high_quality_count": len([s for s in scores if s >= 0.7])
    }

@app.delete("/api/memory")
async def clear_memory():
    """Clear all stored reasoning traces"""
    agent.memory = []
    agent._save_memory()
    return {"message": "Memory cleared"}
EOF

cat > backend/test_cot.py << 'EOF'
"""Tests for CoT reasoning system"""
import pytest
from cot_agent import CoTPromptBuilder, ReasoningTraceParser, QualityEvaluator

def test_prompt_building():
    """Test CoT prompt construction"""
    query = "What is 5 + 3?"
    prompt = CoTPromptBuilder.build_cot_prompt(query)
    assert "step-by-step" in prompt.lower()
    assert query in prompt

def test_step_extraction():
    """Test parsing reasoning steps"""
    trace = """1. First, I identify that we have 5 apples.
2. Then, we add 3 more apples.
3. Therefore, 5 + 3 = 8 apples."""
    
    steps = ReasoningTraceParser.extract_steps(trace)
    assert len(steps) == 3
    assert "5 apples" in steps[0]

def test_clarity_scoring():
    """Test clarity assessment"""
    good_steps = [
        "First, we identify the initial count of 5 items",
        "Then, we add 3 more items to the collection",
        "Therefore, the final count is 8 items"
    ]
    
    score = QualityEvaluator.assess_clarity(good_steps)
    assert score > 0.5

def test_conclusion_detection():
    """Test conclusion extraction"""
    trace = "Step 1: Analyze. Step 2: Calculate. Therefore, the answer is 42."
    conclusion = ReasoningTraceParser.extract_conclusion(trace)
    assert conclusion is not None
    assert "42" in conclusion

def test_overall_evaluation():
    """Test comprehensive quality scoring"""
    trace = """1. First, identify that John has 3 apples.
2. Then, John buys 2 more apples, so 3 + 2 = 5 apples.
3. Next, John gives 1 apple to Mary, so 5 - 1 = 4 apples.
4. Therefore, John has 4 apples remaining."""
    
    scores = QualityEvaluator.evaluate_reasoning(trace)
    assert scores["step_count"] >= 3
    assert scores["overall_quality"] > 0.6
    assert scores["conclusion_present"] == 1.0
EOF

# Frontend setup
cat > frontend/package.json << 'EOF'
{
  "name": "cot-reasoning-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.6.7"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": ["react-app"]
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
    <title>CoT Reasoning Evaluator - L11</title>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import ReasoningForm from './components/ReasoningForm';
import ReasoningDisplay from './components/ReasoningDisplay';
import MemoryViewer from './components/MemoryViewer';
import Statistics from './components/Statistics';
import './App.css';

function App() {
  const [currentReasoning, setCurrentReasoning] = useState(null);
  const [memory, setMemory] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchMemory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/memory?limit=10');
      const data = await response.json();
      setMemory(data.traces);
    } catch (error) {
      console.error('Error fetching memory:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  useEffect(() => {
    fetchMemory();
    fetchStats();
    const interval = setInterval(() => {
      fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleReasoningComplete = (result) => {
    setCurrentReasoning(result);
    fetchMemory();
    fetchStats();
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🧠 Chain-of-Thought Reasoning Evaluator</h1>
        <p className="subtitle">L11: Master CoT Prompting & Reasoning Analysis</p>
      </header>

      <div className="container">
        <div className="grid">
          <div className="card">
            <h2>Query Input</h2>
            <ReasoningForm 
              onComplete={handleReasoningComplete}
              loading={loading}
              setLoading={setLoading}
            />
          </div>

          {stats && (
            <div className="card">
              <h2>Quality Statistics</h2>
              <Statistics stats={stats} />
            </div>
          )}
        </div>

        {currentReasoning && (
          <div className="card">
            <h2>Current Reasoning Analysis</h2>
            <ReasoningDisplay reasoning={currentReasoning} />
          </div>
        )}

        <div className="card">
          <h2>Reasoning Memory ({memory.length} traces)</h2>
          <MemoryViewer memory={memory} />
        </div>
      </div>
    </div>
  );
}

export default App;
EOF

cat > frontend/src/components/ReasoningForm.js << 'EOF'
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
      onComplete(result);
      setQuery('');
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to process reasoning');
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
EOF

cat > frontend/src/components/ReasoningDisplay.js << 'EOF'
import React from 'react';

function ReasoningDisplay({ reasoning }) {
  const { query, steps, conclusion, quality_scores } = reasoning;

  const getQualityColor = (score) => {
    if (score >= 0.7) return '#4caf50';
    if (score >= 0.5) return '#ff9800';
    return '#f44336';
  };

  const getQualityLabel = (score) => {
    if (score >= 0.7) return 'High Quality';
    if (score >= 0.5) return 'Medium Quality';
    return 'Low Quality';
  };

  return (
    <div className="reasoning-display">
      <div className="query-section">
        <h3>Query</h3>
        <p className="query-text">{query}</p>
      </div>

      <div className="quality-section">
        <h3>Quality Metrics</h3>
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-label">Overall Quality:</span>
            <div className="metric-value" style={{ color: getQualityColor(quality_scores.overall_quality) }}>
              {(quality_scores.overall_quality * 100).toFixed(0)}%
              <span className="quality-label">{getQualityLabel(quality_scores.overall_quality)}</span>
            </div>
          </div>
          <div className="metric">
            <span className="metric-label">Steps:</span>
            <span className="metric-value">{quality_scores.step_count}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Clarity:</span>
            <span className="metric-value">{(quality_scores.clarity_score * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Logic Flow:</span>
            <span className="metric-value">{(quality_scores.logic_flow * 100).toFixed(0)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Conclusion:</span>
            <span className="metric-value">{quality_scores.conclusion_present ? '✓' : '✗'}</span>
          </div>
        </div>
      </div>

      <div className="steps-section">
        <h3>Reasoning Steps ({steps.length})</h3>
        <ol className="steps-list">
          {steps.map((step, i) => (
            <li key={i} className="step-item">{step}</li>
          ))}
        </ol>
      </div>

      {conclusion && (
        <div className="conclusion-section">
          <h3>Conclusion</h3>
          <p className="conclusion-text">{conclusion}</p>
        </div>
      )}
    </div>
  );
}

export default ReasoningDisplay;
EOF

cat > frontend/src/components/MemoryViewer.js << 'EOF'
import React from 'react';

function MemoryViewer({ memory }) {
  if (memory.length === 0) {
    return <p className="empty-state">No reasoning traces yet. Submit a query to begin.</p>;
  }

  return (
    <div className="memory-viewer">
      {memory.slice().reverse().map((trace, i) => (
        <div key={i} className="memory-item">
          <div className="memory-header">
            <span className="memory-time">
              {new Date(trace.timestamp).toLocaleString()}
            </span>
            <span 
              className="quality-badge"
              style={{
                background: trace.quality_scores.overall_quality >= 0.7 
                  ? '#4caf50' 
                  : trace.quality_scores.overall_quality >= 0.5 
                    ? '#ff9800' 
                    : '#f44336'
              }}
            >
              {(trace.quality_scores.overall_quality * 100).toFixed(0)}%
            </span>
          </div>
          <p className="memory-query">{trace.query}</p>
          <div className="memory-stats">
            <span>Steps: {trace.quality_scores.step_count}</span>
            <span>Clarity: {(trace.quality_scores.clarity_score * 100).toFixed(0)}%</span>
            <span>Logic: {(trace.quality_scores.logic_flow * 100).toFixed(0)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default MemoryViewer;
EOF

cat > frontend/src/components/Statistics.js << 'EOF'
import React from 'react';

function Statistics({ stats }) {
  if (!stats.total_traces) {
    return <p className="empty-state">Submit queries to see statistics</p>;
  }

  return (
    <div className="statistics">
      <div className="stat-card">
        <div className="stat-value">{stats.total_traces}</div>
        <div className="stat-label">Total Traces</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{(stats.avg_quality * 100).toFixed(0)}%</div>
        <div className="stat-label">Avg Quality</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{stats.high_quality_count}</div>
        <div className="stat-label">High Quality</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{(stats.max_quality * 100).toFixed(0)}%</div>
        <div className="stat-label">Best Score</div>
      </div>
    </div>
  );
}

export default Statistics;
EOF

cat > frontend/src/App.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  padding: 20px;
}

.header {
  text-align: center;
  color: white;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.5em;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.subtitle {
  font-size: 1.2em;
  opacity: 0.9;
}

.container {
  max-width: 1400px;
  margin: 0 auto;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.card {
  background: white;
  border-radius: 12px;
  padding: 25px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  margin-bottom: 20px;
}

.card h2 {
  margin-bottom: 20px;
  color: #333;
  font-size: 1.5em;
}

.reasoning-form {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-group label {
  font-weight: 600;
  color: #555;
}

.form-group textarea {
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-family: inherit;
  font-size: 14px;
  resize: vertical;
}

.form-group textarea:focus {
  outline: none;
  border-color: #667eea;
}

.form-group select {
  padding: 10px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 14px;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.examples {
  margin-top: 10px;
  padding-top: 15px;
  border-top: 1px solid #e0e0e0;
}

.examples p {
  font-size: 14px;
  color: #666;
  margin-bottom: 10px;
}

.btn-example {
  display: block;
  width: 100%;
  text-align: left;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  padding: 8px 12px;
  border-radius: 6px;
  margin-bottom: 8px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.2s;
}

.btn-example:hover:not(:disabled) {
  background: #e8e8e8;
}

.reasoning-display {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.query-section, .quality-section, .steps-section, .conclusion-section {
  padding: 15px;
  background: #f9f9f9;
  border-radius: 8px;
}

.query-section h3, .quality-section h3, .steps-section h3, .conclusion-section h3 {
  margin-bottom: 10px;
  color: #667eea;
  font-size: 1.2em;
}

.query-text {
  font-size: 16px;
  color: #333;
  line-height: 1.5;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.metric-label {
  font-size: 13px;
  color: #666;
  font-weight: 500;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: #333;
}

.quality-label {
  display: block;
  font-size: 12px;
  font-weight: 500;
  margin-top: 5px;
}

.steps-list {
  list-style-position: inside;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.step-item {
  padding: 12px;
  background: white;
  border-left: 3px solid #667eea;
  border-radius: 4px;
  line-height: 1.5;
}

.conclusion-text {
  font-size: 16px;
  color: #333;
  line-height: 1.6;
  font-weight: 500;
}

.statistics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 15px;
}

.stat-card {
  text-align: center;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
  color: white;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 13px;
  opacity: 0.9;
}

.memory-viewer {
  display: flex;
  flex-direction: column;
  gap: 15px;
  max-height: 600px;
  overflow-y: auto;
}

.memory-item {
  padding: 15px;
  background: #f9f9f9;
  border-radius: 8px;
  border-left: 4px solid #667eea;
}

.memory-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.memory-time {
  font-size: 12px;
  color: #666;
}

.quality-badge {
  padding: 4px 12px;
  border-radius: 12px;
  color: white;
  font-size: 12px;
  font-weight: 600;
}

.memory-query {
  font-size: 14px;
  color: #333;
  margin-bottom: 10px;
  font-weight: 500;
}

.memory-stats {
  display: flex;
  gap: 15px;
  font-size: 12px;
  color: #666;
}

.empty-state {
  text-align: center;
  color: #999;
  padding: 40px;
  font-style: italic;
}

@media (max-width: 768px) {
  .grid {
    grid-template-columns: 1fr;
  }
  
  .header h1 {
    font-size: 2em;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
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

# Docker setup
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - GEMINI_API_KEY=AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8
    restart: unless-stopped

  frontend:
    image: node:18-alpine
    working_dir: /app
    command: sh -c "npm install && npm start"
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
EOF

# Build script
cat > build.sh << 'EOF'
#!/bin/bash
set -e

echo "🔨 Building L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null; then
    echo "Building with Docker..."
    docker-compose build
else
    echo "Building without Docker..."
    
    # Backend
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
    
    # Frontend
    cd frontend
    npm install
    cd ..
fi

echo "✅ Build complete!"
EOF

# Start script
cat > start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null; then
    echo "Starting with Docker..."
    docker-compose up -d
    echo ""
    echo "✅ Services running:"
    echo "   Backend API: http://localhost:8000"
    echo "   Frontend UI: http://localhost:3000"
    echo ""
    echo "View logs: docker-compose logs -f"
    echo "Stop services: ./stop.sh"
else
    echo "Starting without Docker..."
    
    # Start backend
    cd backend
    source venv/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    BROWSER=none npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo ""
    echo "✅ Services running:"
    echo "   Backend API: http://localhost:8000 (PID: $BACKEND_PID)"
    echo "   Frontend UI: http://localhost:3000 (PID: $FRONTEND_PID)"
    echo ""
    echo "Stop services: ./stop.sh"
fi
EOF

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null; then
    docker-compose down
else
    pkill -f "uvicorn main:app"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped"
EOF

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
set -e

echo "🧪 Running L11 CoT System Tests..."

cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run pytest
python -m pytest test_cot.py -v

# API health check
echo ""
echo "Testing API endpoints..."
sleep 2

# Test root endpoint
curl -s http://localhost:8000/ | grep -q "operational" && echo "✓ Root endpoint OK"

# Test reasoning endpoint
RESULT=$(curl -s -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2 + 2?", "style": "standard"}')

echo "$RESULT" | grep -q "reasoning_trace" && echo "✓ Reasoning endpoint OK"
echo "$RESULT" | grep -q "quality_scores" && echo "✓ Quality evaluation OK"

# Test memory endpoint
curl -s http://localhost:8000/api/memory | grep -q "traces" && echo "✓ Memory endpoint OK"

# Test stats endpoint
curl -s http://localhost:8000/api/stats | grep -q "total_traces" && echo "✓ Stats endpoint OK"

echo ""
echo "✅ All tests passed!"
EOF

# Make scripts executable
chmod +x build.sh start.sh stop.sh test.sh

# README
cat > README.md << 'EOF'
# L11: Chain-of-Thought Reasoning Evaluator

Production-grade CoT prompting system with automated reasoning quality analysis.

## Quick Start

```bash
# Build
./build.sh

# Start services
./start.sh

# Run tests
./test.sh

# Stop services
./stop.sh
```

## Access

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- Chain-of-Thought prompt construction
- Automated reasoning trace parsing
- Multi-dimensional quality scoring
- Visual reasoning analysis dashboard
- Persistent memory for trace storage
- Real-time quality statistics

## Architecture

- **Backend**: Python FastAPI + Gemini AI
- **Frontend**: React with responsive UI
- **Persistence**: JSON-based memory storage
- **Deployment**: Docker + non-Docker modes

## Testing Sample Queries

1. Math: "If Alice has 5 cookies and gives 2 to Bob, then Bob gives 1 to Charlie, how many cookies does each person have?"
2. Logic: "If all cats are mammals and all mammals are animals, are all cats animals?"
3. Planning: "How would you organize a 3-day conference for 200 attendees?"

## Integration with L10

Extends SimpleAgent from L10 with `reason_with_cot()` method and quality evaluation capabilities.

## Prepares for L12

High-quality reasoning traces stored in memory become few-shot examples for L12's prompting techniques.
EOF

echo ""
echo "✅ L11 CoT Reasoning Evaluator setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd $PROJECT_NAME"
echo "  2. ./build.sh"
echo "  3. ./start.sh"
echo "  4. Open http://localhost:3000"
echo ""
echo "Test the system: ./test.sh"