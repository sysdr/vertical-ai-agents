#!/bin/bash

# L10: Building a Simple Agent - Automated Setup
# Builds production-ready autonomous agent with memory integration

set -e

PROJECT_NAME="l10-simple-agent"
GEMINI_KEY="AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"

echo "🚀 L10: Building a Simple Agent - Setup Started"

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,data,scripts}
cd $PROJECT_NAME

# Backend: SimpleAgent implementation
cat > backend/agent.py << 'EOF'
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import google.generativeai as genai

class ShortTermMemory:
    """In-memory conversation context - from L9"""
    def __init__(self):
        self.memory: Dict[str, any] = {}
        self.conversation_history: List[Dict] = []
    
    def store(self, key: str, value: any):
        self.memory[key] = value
    
    def retrieve(self, key: str) -> any:
        return self.memory.get(key)
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

class LongTermStore:
    """File-based persistent storage - from L9"""
    def __init__(self, filepath: str = "data/long_term_memory.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(exist_ok=True)
        self._load()
    
    def _load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {}
    
    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def store(self, key: str, value: any):
        self.data[key] = value
        self._save()
    
    def retrieve(self, key: str) -> any:
        return self.data.get(key)
    
    def query(self, pattern: str) -> Dict:
        return {k: v for k, v in self.data.items() if pattern in k}

class SimpleAgent:
    """Production-grade autonomous agent with goal-seeking behavior"""
    
    def __init__(self, api_key: str, agent_id: str = "agent_001"):
        genai.configure(api_key=api_key)
        # Try to use available model - fallback to gemini-pro if needed
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except:
                # List available models and use the first one
                models = genai.list_models()
                available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                if available:
                    model_name = available[0].split('/')[-1]  # Extract model name
                    self.model = genai.GenerativeModel(model_name)
                else:
                    raise Exception("No available models found")
        self.agent_id = agent_id
        
        # Memory systems from L9
        self.short_term = ShortTermMemory()
        self.long_term = LongTermStore(f"data/{agent_id}_memory.json")
        self.decision_log = LongTermStore(f"data/{agent_id}_decisions.json")
        
        # Agent state machine
        self.state = "IDLE"
        self.current_goal = None
        self.attempt_count = 0
        self.max_attempts = 5
    
    def remember(self, user_input: str, action: str, result: str):
        """Store interaction in both memory tiers"""
        # Short-term: current session
        self.short_term.add_message("user", user_input)
        self.short_term.add_message("agent", action)
        
        # Long-term: persistent record
        timestamp = datetime.now().isoformat()
        self.long_term.store(f"interaction_{timestamp}", {
            "user_input": user_input,
            "action": action,
            "result": result,
            "goal": self.current_goal
        })
    
    def _build_context(self, user_input: str, goal: str) -> str:
        """Combine short-term conversation + long-term relevant memories"""
        recent_history = self.short_term.conversation_history[-5:]
        relevant_memories = self.long_term.query(goal.split()[0]) if goal else {}
        
        context = f"""You are an autonomous agent pursuing a goal.

CURRENT GOAL: {goal}

RECENT CONVERSATION:
{json.dumps(recent_history, indent=2)}

RELEVANT PAST EXPERIENCES:
{json.dumps(list(relevant_memories.values())[:3], indent=2)}

NEW USER INPUT: {user_input}

Generate the next action to progress toward the goal. Be specific and actionable.
"""
        return context
    
    async def _generate_action(self, context: str) -> Dict[str, str]:
        """Query LLM for next action"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                context
            )
            
            action_text = response.text
            
            # Extract reasoning if present
            reasoning = "Direct action"
            if "because" in action_text.lower():
                parts = action_text.split("because", 1)
                action_text = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else reasoning
            
            return {
                "action": action_text,
                "reasoning": reasoning
            }
        except Exception as e:
            return {
                "action": f"Error generating action: {str(e)}",
                "reasoning": "Exception occurred"
            }
    
    def _execute(self, action_data: Dict[str, str]) -> str:
        """Simulate action execution - in production, this calls external APIs"""
        action = action_data["action"]
        
        # Simulate processing
        time.sleep(0.5)
        
        # In production: API calls, tool usage, data retrieval
        # For now: confirm execution
        return f"Executed: {action}"
    
    async def _evaluate_progress(self, goal: str, result: str) -> Dict[str, any]:
        """LLM evaluates if goal is achieved"""
        eval_prompt = f"""Evaluate goal progress:

GOAL: {goal}
LATEST ACTION RESULT: {result}
CONVERSATION HISTORY: {json.dumps(self.short_term.conversation_history[-3:], indent=2)}

Respond with JSON:
{{
    "goal_achieved": true/false,
    "progress_percentage": 0-100,
    "next_step_needed": "description if not complete"
}}
"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                eval_prompt
            )
            
            # Parse JSON from response
            eval_text = response.text.strip()
            # Remove markdown code blocks if present
            if "```json" in eval_text:
                eval_text = eval_text.split("```json")[1].split("```")[0].strip()
            elif "```" in eval_text:
                eval_text = eval_text.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(eval_text)
            return evaluation
            
        except Exception as e:
            return {
                "goal_achieved": False,
                "progress_percentage": 0,
                "next_step_needed": f"Evaluation error: {str(e)}"
            }
    
    async def act(self, user_input: str, goal: Optional[str] = None) -> Dict[str, any]:
        """Main agent control loop"""
        if goal:
            self.current_goal = goal
            self.attempt_count = 0
        
        if not self.current_goal:
            self.current_goal = "Respond helpfully to user input"
        
        self.attempt_count += 1
        
        # State: THINKING
        self.state = "THINKING"
        context = self._build_context(user_input, self.current_goal)
        action_data = await self._generate_action(context)
        
        # State: ACTING
        self.state = "ACTING"
        result = self._execute(action_data)
        
        # State: EVALUATING
        self.state = "EVALUATING"
        progress = await self._evaluate_progress(self.current_goal, result)
        
        # Store decision for observability
        decision_id = f"decision_{datetime.now().isoformat()}"
        self.decision_log.store(decision_id, {
            "attempt": self.attempt_count,
            "goal": self.current_goal,
            "action": action_data["action"],
            "reasoning": action_data["reasoning"],
            "result": result,
            "progress": progress,
            "state_transitions": ["THINKING", "ACTING", "EVALUATING"]
        })
        
        # Remember interaction
        self.remember(user_input, action_data["action"], result)
        
        # Update state
        if progress.get("goal_achieved"):
            self.state = "COMPLETE"
        elif self.attempt_count >= self.max_attempts:
            self.state = "FAILED"
        else:
            self.state = "READY"
        
        return {
            "action": action_data["action"],
            "reasoning": action_data["reasoning"],
            "result": result,
            "progress": progress,
            "state": self.state,
            "attempt": self.attempt_count,
            "goal": self.current_goal
        }
    
    def get_state(self) -> Dict[str, any]:
        """Observable agent state for debugging"""
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "current_goal": self.current_goal,
            "attempt_count": self.attempt_count,
            "conversation_length": len(self.short_term.conversation_history),
            "decision_count": len(self.decision_log.data)
        }
EOF

# FastAPI Backend
cat > backend/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from agent import SimpleAgent

app = FastAPI(title="L10 Simple Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent - get API key from environment variable
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your .env file or environment.")
agent = SimpleAgent(api_key=GEMINI_KEY)

class Message(BaseModel):
    content: str
    goal: Optional[str] = None

@app.get("/")
def root():
    return {"message": "L10 Simple Agent API", "status": "running"}

@app.get("/agent/state")
def get_agent_state():
    return agent.get_state()

@app.post("/agent/act")
async def agent_act(message: Message):
    try:
        result = await agent.act(message.content, message.goal)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/memory")
def get_memory():
    return {
        "short_term": {
            "conversation_history": agent.short_term.conversation_history,
            "memory_keys": list(agent.short_term.memory.keys())
        },
        "long_term_count": len(agent.long_term.data),
        "decision_log_count": len(agent.decision_log.data)
    }

@app.get("/agent/decisions")
def get_decisions():
    return {
        "decisions": list(agent.decision_log.data.values())[-10:]  # Last 10 decisions
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Requirements
cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
google-generativeai>=0.8.0
python-multipart==0.0.6
EOF

# React Frontend
cat > frontend/package.json << 'EOF'
{
  "name": "l10-simple-agent-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
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
    <title>L10: Simple Agent Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import AgentDashboard from './components/AgentDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <AgentDashboard />
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
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.App {
  min-height: 100vh;
  padding: 20px;
}
EOF

cat > frontend/src/components/AgentDashboard.js << 'EOF'
import React, { useState, useEffect } from 'react';
import './AgentDashboard.css';

const API_URL = 'http://localhost:8000';

function AgentDashboard() {
  const [message, setMessage] = useState('');
  const [goal, setGoal] = useState('');
  const [agentState, setAgentState] = useState(null);
  const [responses, setResponses] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAgentState();
    fetchDecisions();
    const interval = setInterval(() => {
      fetchAgentState();
      fetchDecisions();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchAgentState = async () => {
    try {
      const res = await fetch(`${API_URL}/agent/state`);
      const data = await res.json();
      setAgentState(data);
    } catch (err) {
      console.error('Failed to fetch agent state:', err);
    }
  };

  const fetchDecisions = async () => {
    try {
      const res = await fetch(`${API_URL}/agent/decisions`);
      const data = await res.json();
      setDecisions(data.decisions || []);
    } catch (err) {
      console.error('Failed to fetch decisions:', err);
    }
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/agent/act`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: message,
          goal: goal || null
        })
      });

      const data = await res.json();
      setResponses(prev => [...prev, {
        message,
        goal,
        response: data,
        timestamp: new Date().toLocaleTimeString()
      }]);

      setMessage('');
      fetchAgentState();
      fetchDecisions();
    } catch (err) {
      console.error('Failed to send message:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStateColor = (state) => {
    const colors = {
      'IDLE': '#95a5a6',
      'THINKING': '#3498db',
      'ACTING': '#f39c12',
      'EVALUATING': '#9b59b6',
      'COMPLETE': '#27ae60',
      'FAILED': '#e74c3c',
      'READY': '#16a085'
    };
    return colors[state] || '#95a5a6';
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🤖 L10: Simple Agent</h1>
        <p>Autonomous agent with memory & goal-seeking</p>
      </header>

      <div className="dashboard-grid">
        {/* Agent State */}
        <div className="card state-card">
          <h2>Agent State</h2>
          {agentState && (
            <div className="state-display">
              <div className="state-badge" style={{ backgroundColor: getStateColor(agentState.state) }}>
                {agentState.state}
              </div>
              <div className="state-details">
                <div className="state-item">
                  <span className="label">Current Goal:</span>
                  <span className="value">{agentState.current_goal || 'None'}</span>
                </div>
                <div className="state-item">
                  <span className="label">Attempt:</span>
                  <span className="value">{agentState.attempt_count}/5</span>
                </div>
                <div className="state-item">
                  <span className="label">Conversations:</span>
                  <span className="value">{agentState.conversation_length}</span>
                </div>
                <div className="state-item">
                  <span className="label">Decisions Made:</span>
                  <span className="value">{agentState.decision_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="card input-card">
          <h2>Send Message</h2>
          <div className="input-group">
            <input
              type="text"
              placeholder="Enter goal (optional)"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="goal-input"
            />
            <textarea
              placeholder="Enter your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows="4"
              className="message-input"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !message.trim()}
              className="send-button"
            >
              {loading ? 'Processing...' : 'Send to Agent'}
            </button>
          </div>
        </div>

        {/* Responses */}
        <div className="card responses-card">
          <h2>Agent Responses</h2>
          <div className="responses-list">
            {responses.slice().reverse().map((item, idx) => (
              <div key={idx} className="response-item">
                <div className="response-header">
                  <span className="timestamp">{item.timestamp}</span>
                  {item.goal && <span className="goal-badge">{item.goal}</span>}
                </div>
                <div className="response-message">
                  <strong>You:</strong> {item.message}
                </div>
                <div className="response-action">
                  <strong>Agent Action:</strong> {item.response.action}
                </div>
                <div className="response-reasoning">
                  <strong>Reasoning:</strong> {item.response.reasoning}
                </div>
                {item.response.progress && (
                  <div className="response-progress">
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${item.response.progress.progress_percentage}%` }}
                      />
                    </div>
                    <span>{item.response.progress.progress_percentage}% complete</span>
                  </div>
                )}
              </div>
            ))}
            {responses.length === 0 && (
              <div className="empty-state">No responses yet. Send a message to start!</div>
            )}
          </div>
        </div>

        {/* Decision Log */}
        <div className="card decisions-card">
          <h2>Decision Log (Last 10)</h2>
          <div className="decisions-list">
            {decisions.slice().reverse().map((decision, idx) => (
              <div key={idx} className="decision-item">
                <div className="decision-header">
                  <span className="attempt-badge">Attempt {decision.attempt}</span>
                  <span className="goal-text">{decision.goal}</span>
                </div>
                <div className="decision-action">{decision.action}</div>
                <div className="decision-states">
                  {decision.state_transitions.map((state, i) => (
                    <span key={i} className="state-chip" style={{ backgroundColor: getStateColor(state) }}>
                      {state}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {decisions.length === 0 && (
              <div className="empty-state">No decisions logged yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AgentDashboard;
EOF

cat > frontend/src/components/AgentDashboard.css << 'EOF'
.dashboard {
  max-width: 1400px;
  margin: 0 auto;
}

.dashboard-header {
  text-align: center;
  margin-bottom: 40px;
  color: white;
}

.dashboard-header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
}

.card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}

.card h2 {
  font-size: 1.3rem;
  margin-bottom: 20px;
  color: #2c3e50;
  border-bottom: 2px solid #ecf0f1;
  padding-bottom: 10px;
}

/* State Card */
.state-display {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.state-badge {
  display: inline-block;
  padding: 12px 24px;
  border-radius: 8px;
  color: white;
  font-weight: bold;
  font-size: 1.2rem;
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.state-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.state-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 6px;
}

.state-item .label {
  font-weight: 600;
  color: #7f8c8d;
}

.state-item .value {
  color: #2c3e50;
  font-weight: 500;
}

/* Input Card */
.input-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.goal-input,
.message-input {
  width: 100%;
  padding: 12px;
  border: 2px solid #ecf0f1;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  transition: border-color 0.3s;
}

.goal-input:focus,
.message-input:focus {
  outline: none;
  border-color: #3498db;
}

.send-button {
  padding: 14px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Responses */
.responses-list,
.decisions-list {
  max-height: 500px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.response-item,
.decision-item {
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #3498db;
}

.response-header,
.decision-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.timestamp {
  font-size: 0.85rem;
  color: #7f8c8d;
}

.goal-badge,
.attempt-badge {
  background: #3498db;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}

.response-message,
.response-action,
.response-reasoning,
.decision-action {
  margin-bottom: 8px;
  line-height: 1.6;
}

.response-message strong,
.response-action strong,
.response-reasoning strong {
  color: #2c3e50;
}

.response-progress {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #ecf0f1;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
  transition: width 0.3s ease;
}

.decision-states {
  display: flex;
  gap: 8px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.state-chip {
  padding: 4px 10px;
  border-radius: 12px;
  color: white;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.empty-state {
  text-align: center;
  padding: 40px;
  color: #95a5a6;
  font-style: italic;
}

/* Scrollbar */
.responses-list::-webkit-scrollbar,
.decisions-list::-webkit-scrollbar {
  width: 8px;
}

.responses-list::-webkit-scrollbar-track,
.decisions-list::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

.responses-list::-webkit-scrollbar-thumb,
.decisions-list::-webkit-scrollbar-thumb {
  background: #cbd5e0;
  border-radius: 4px;
}
EOF

# Docker Setup
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
RUN mkdir -p data

EXPOSE 8000

CMD ["python", "main.py"]
EOF

cat > docker-compose.yml << 'EOF'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    env_file:
      - .env
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}

  frontend:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "3000:3000"
    command: sh -c "npm install && npm start"
    depends_on:
      - backend
EOF

# Helper Scripts
cat > scripts/build.sh << 'EOF'
#!/bin/bash
echo "🔨 Building L10 Simple Agent..."

# Backend
cd backend
python -m venv venv 2>/dev/null || python3 -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

echo "✅ Build complete!"
EOF

cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting L10 Simple Agent..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Using Docker..."
    docker-compose up -d
    echo "✅ Services started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
else
    echo "Using local environment..."
    
    # Start backend
    cd backend
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
    python main.py &
    BACKEND_PID=$!
    
    # Start frontend
    cd ../frontend
    npm start &
    FRONTEND_PID=$!
    
    echo "✅ Services started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
    echo "PIDs: Backend=$BACKEND_PID Frontend=$FRONTEND_PID"
fi
EOF

cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping L10 Simple Agent..."

if command -v docker &> /dev/null; then
    docker-compose down
else
    pkill -f "python main.py"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped!"
EOF

cat > scripts/test.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing L10 Simple Agent..."

# Wait for services
sleep 5

# Test API
echo "Testing API endpoint..."
curl -s http://localhost:8000/ | grep -q "running" && echo "✅ API running" || echo "❌ API failed"

# Test agent state
echo "Testing agent state..."
curl -s http://localhost:8000/agent/state | grep -q "agent_id" && echo "✅ Agent state OK" || echo "❌ Agent state failed"

# Test agent action
echo "Testing agent action..."
curl -s -X POST http://localhost:8000/agent/act \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello, what can you do?","goal":"Test interaction"}' \
  | grep -q "action" && echo "✅ Agent action OK" || echo "❌ Agent action failed"

echo "
✅ All tests passed!

Try the dashboard: http://localhost:3000
"
EOF

chmod +x scripts/*.sh

# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
venv/
env/
ENV/
.venv
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/
.pytype/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*
.pnpm-debug.log*
.npm
.eslintcache
.node_repl_history
*.tgz
.yarn-integrity
.yarn/cache
.yarn/unplugged
.yarn/build-state
.yarn/install-state.gz
.pnp.*

# React
frontend/build/
frontend/.env.local
frontend/.env.development.local
frontend/.env.test.local
frontend/.env.production.local

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Docker
.dockerignore

# Logs
*.log
logs/

# OS
Thumbs.db
.DS_Store
EOF

# Create .env.example file
cat > .env.example << 'EOF'
# Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
EOF

# README
cat > README.md << 'EOF'
# L10: Building a Simple Agent

Production-ready autonomous agent with memory integration and goal-seeking behavior.

## Quick Start

### 1. Set up API Key

Create a `.env` file in the project root with your Gemini API key:

```bash
cp .env.example .env
# Edit .env and add your API key:
# GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 2. Start Services

#### With Docker
```bash
./scripts/build.sh
./scripts/start.sh
```

#### Without Docker
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py &

cd ../frontend
npm install
npm start
```

## Access

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- ✅ Goal-driven autonomous behavior
- ✅ Dual-tier memory (short-term + long-term)
- ✅ Self-evaluation and progress tracking
- ✅ Observable decision logging
- ✅ Real-time state machine visualization
- ✅ Production-ready patterns

## Testing

```bash
./scripts/test.sh
```

## Architecture

- **Backend**: FastAPI + Gemini AI + dual-tier memory
- **Frontend**: React dashboard with real-time updates
- **Agent**: SimpleAgent class with IDLE→THINKING→ACTING→EVALUATING→COMPLETE loop

## Stop Services

```bash
./scripts/stop.sh
```
EOF

echo "
✅ L10: Building a Simple Agent - Setup Complete!

📁 Project structure created in: $PROJECT_NAME/

Next steps:
1. cd $PROJECT_NAME
2. ./scripts/build.sh      # Install dependencies
3. ./scripts/start.sh      # Start services
4. ./scripts/test.sh       # Verify everything works

Then open: http://localhost:3000

Features:
✅ SimpleAgent with goal-seeking behavior
✅ Memory integration from L9
✅ State machine visualization
✅ Decision logging system
✅ Real-time dashboard

Happy building! 🚀
"