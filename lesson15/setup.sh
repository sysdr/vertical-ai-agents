#!/bin/bash

# L15: Basic Conversational Agent - Automated Setup Script
set -e

PROJECT_NAME="conversational-agent"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

echo "=========================================="
echo "L15: Basic Conversational Agent Setup"
echo "=========================================="

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,scripts,data,tests}
cd $PROJECT_NAME

# Backend files
cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.0
google-generativeai==0.8.3
python-dotenv==1.0.1
aiosqlite==0.20.0
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
EOF

cat > backend/.env << EOF
GEMINI_API_KEY=$GEMINI_API_KEY
DATABASE_PATH=../data/conversations.db
LOG_LEVEL=INFO
EOF

cat > backend/models.py << 'EOF'
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class ConversationState(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    GOAL_SEEKING = "goal_seeking"
    COMPLETED = "completed"

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    
    def dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0)
        )

@dataclass
class Goal:
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completion_criteria: Optional[str] = None
    
    def dict(self) -> dict:
        return {
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "completed": self.completed,
            "completion_criteria": self.completion_criteria
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed=data.get("completed", False),
            completion_criteria=data.get("completion_criteria")
        )

@dataclass
class ConversationStateModel:
    user_id: str
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    active_goals: List[Goal] = field(default_factory=list)
    state: ConversationState = ConversationState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "messages": [m.dict() for m in self.messages],
            "active_goals": [g.dict() for g in self.active_goals],
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "total_tokens": self.total_tokens
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            user_id=data["user_id"],
            conversation_id=data["conversation_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            active_goals=[Goal.from_dict(g) for g in data.get("active_goals", [])],
            state=ConversationState(data.get("state", "initializing")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            total_tokens=data.get("total_tokens", 0)
        )
    
    def add_message(self, role: str, content: str, token_count: int = 0):
        msg = Message(role=role, content=content, token_count=token_count)
        self.messages.append(msg)
        self.total_tokens += token_count
        self.updated_at = datetime.now()
        if len(self.messages) == 1:
            self.state = ConversationState.ACTIVE
    
    def add_goal(self, description: str, criteria: Optional[str] = None):
        goal = Goal(description=description, completion_criteria=criteria)
        self.active_goals.append(goal)
        self.state = ConversationState.GOAL_SEEKING
        self.updated_at = datetime.now()
    
    def complete_goal(self, goal_description: str):
        for goal in self.active_goals:
            if goal.description == goal_description:
                goal.completed = True
                self.updated_at = datetime.now()
                break
        if all(g.completed for g in self.active_goals) and self.active_goals:
            self.state = ConversationState.COMPLETED
EOF

cat > backend/memory_manager.py << 'EOF'
import aiosqlite
import json
from typing import Optional, List
from models import ConversationStateModel
from datetime import datetime
import os

class MemoryManager:
    def __init__(self, db_path: str = "../data/conversations.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT UNIQUE NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversations(conversation_id)")
            await db.commit()
    
    async def save_state(self, state: ConversationStateModel):
        async with aiosqlite.connect(self.db_path) as db:
            state_json = json.dumps(state.to_dict())
            await db.execute("""
                INSERT OR REPLACE INTO conversations 
                (user_id, conversation_id, state_json, created_at, updated_at, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (state.user_id, state.conversation_id, state_json,
                  state.created_at.isoformat(), state.updated_at.isoformat(), state.total_tokens))
            await db.commit()
    
    async def load_state(self, conversation_id: str) -> Optional[ConversationStateModel]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT state_json FROM conversations WHERE conversation_id = ?", 
                                (conversation_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ConversationStateModel.from_dict(json.loads(row[0]))
                return None
    
    async def list_conversations(self, user_id: str) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT conversation_id, created_at, updated_at, total_tokens
                FROM conversations WHERE user_id = ? ORDER BY updated_at DESC
            """, (user_id,)) as cursor:
                rows = await cursor.fetchall()
                return [{"conversation_id": r[0], "created_at": r[1], 
                        "updated_at": r[2], "total_tokens": r[3]} for r in rows]
EOF

cat > backend/goal_tracker.py << 'EOF'
from models import Goal, ConversationStateModel
import google.generativeai as genai

class GoalTracker:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def evaluate_goal_completion(self, goal: Goal, conversation_history: str) -> bool:
        if goal.completed:
            return True
        prompt = f"""Analyze if this goal has been achieved:
Goal: {goal.description}
Criteria: {goal.completion_criteria or 'User satisfaction'}
Conversation: {conversation_history}
Has the goal been achieved? Respond with only 'YES' or 'NO' and brief explanation."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().lower().startswith('yes')
        except:
            return False
    
    async def update_goals(self, state: ConversationStateModel) -> bool:
        if not state.active_goals:
            return False
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in state.messages[-10:]])
        any_completed = False
        for goal in state.active_goals:
            if not goal.completed:
                if await self.evaluate_goal_completion(goal, history):
                    state.complete_goal(goal.description)
                    any_completed = True
        return any_completed
    
    def get_active_goals_context(self, state: ConversationStateModel) -> str:
        active = [g for g in state.active_goals if not g.completed]
        if not active:
            return ""
        return "\nActive Goals:\n" + "\n".join([f"- {g.description}" for g in active])
EOF

cat > backend/gemini_client.py << 'EOF'
import google.generativeai as genai
from typing import List
from models import Message
import time

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.max_retries = 3
        self.base_delay = 1.0
    
    async def generate_response(self, messages: List[Message], system_context: str = "") -> tuple[str, int]:
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[:-1]])
        current = messages[-1].content if messages else ""
        prompt = f"""{system_context}
Conversation History:
{history}
User: {current}
Assistant:"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                tokens = len(text.split()) * 2  # Rough estimate
                return text, tokens
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    return f"Error: {str(e)}", 0
EOF

cat > backend/conversation_engine.py << 'EOF'
from models import ConversationStateModel, ConversationState
from memory_manager import MemoryManager
from goal_tracker import GoalTracker
from gemini_client import GeminiClient
from typing import Optional
import uuid

class ConversationEngine:
    def __init__(self, api_key: str, db_path: str = "../data/conversations.db"):
        self.memory = MemoryManager(db_path)
        self.goal_tracker = GoalTracker(api_key)
        self.llm = GeminiClient(api_key)
    
    async def initialize(self):
        await self.memory.initialize()
    
    async def create_conversation(self, user_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        state = ConversationStateModel(user_id=user_id, conversation_id=conversation_id)
        await self.memory.save_state(state)
        return conversation_id
    
    async def process_message(self, conversation_id: str, user_message: str) -> dict:
        state = await self.memory.load_state(conversation_id)
        if not state:
            return {"error": "Conversation not found"}
        
        # Add user message
        state.add_message("user", user_message)
        
        # Check for goal commands
        if user_message.lower().startswith("/goal "):
            goal_text = user_message[6:].strip()
            state.add_goal(goal_text)
            response_text = f"Goal set: {goal_text}"
            state.add_message("assistant", response_text)
        else:
            # Build context with goals
            goals_context = self.goal_tracker.get_active_goals_context(state)
            system_context = f"""You are a helpful AI assistant with persistent memory.
You remember previous conversations and work towards user goals.
{goals_context}
Respond naturally and helpfully."""
            
            # Generate response
            response_text, tokens = await self.llm.generate_response(state.messages, system_context)
            state.add_message("assistant", response_text, tokens)
            
            # Update goals
            await self.goal_tracker.update_goals(state)
        
        # Save state
        await self.memory.save_state(state)
        
        return {
            "response": response_text,
            "state": state.state.value,
            "active_goals": len([g for g in state.active_goals if not g.completed]),
            "total_messages": len(state.messages),
            "total_tokens": state.total_tokens
        }
EOF

cat > backend/api.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from conversation_engine import ConversationEngine
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Conversational Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = os.getenv("DATABASE_PATH", "../data/conversations.db")

engine = ConversationEngine(API_KEY, DB_PATH)

class CreateConversationRequest(BaseModel):
    user_id: str

class MessageRequest(BaseModel):
    conversation_id: str
    message: str

@app.on_event("startup")
async def startup():
    await engine.initialize()

@app.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    conversation_id = await engine.create_conversation(req.user_id)
    return {"conversation_id": conversation_id}

@app.post("/messages")
async def send_message(req: MessageRequest):
    result = await engine.process_message(req.conversation_id, req.message)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/conversations/{conversation_id}/history")
async def get_history(conversation_id: str):
    state = await engine.memory.load_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "messages": [m.dict() for m in state.messages],
        "goals": [g.dict() for g in state.active_goals],
        "state": state.state.value
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF

cat > backend/cli.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from conversation_engine import ConversationEngine
from dotenv import load_dotenv
import os
import sys

load_dotenv()

async def main():
    API_KEY = os.getenv("GEMINI_API_KEY")
    engine = ConversationEngine(API_KEY)
    await engine.initialize()
    
    print("=== Conversational Agent CLI ===")
    print("Commands: /goal <description>, /quit")
    
    user_id = input("Enter your user ID: ").strip()
    conversation_id = await engine.create_conversation(user_id)
    print(f"Conversation started: {conversation_id}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
            
            result = await engine.process_message(conversation_id, user_input)
            print(f"\nAssistant: {result['response']}\n")
            print(f"[State: {result['state']}, Goals: {result['active_goals']}, Tokens: {result['total_tokens']}]\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x backend/cli.py

# Frontend
cat > frontend/package.json << 'EOF'
{
  "name": "conversational-agent-dashboard",
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
  "devDependencies": {
    "react-scripts": "5.0.1"
  },
  "browserslist": {
    "production": [">0.2%", "not dead"],
    "development": ["last 1 chrome version"]
  }
}
EOF

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Conversational Agent Dashboard</title>
</head>
<body>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
EOF

cat > frontend/src/index.css << 'EOF'
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ConversationView from './components/ConversationView';

const API_URL = 'http://localhost:8000';

function App() {
  const [userId, setUserId] = useState('');
  const [conversationId, setConversationId] = useState('');
  const [message, setMessage] = useState('');
  const [history, setHistory] = useState([]);
  const [goals, setGoals] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(false);

  const createConversation = async () => {
    if (!userId.trim()) return;
    try {
      const res = await axios.post(`${API_URL}/conversations`, { user_id: userId });
      setConversationId(res.data.conversation_id);
      setHistory([]);
      setGoals([]);
    } catch (error) {
      console.error('Error creating conversation:', error);
    }
  };

  const sendMessage = async () => {
    if (!message.trim() || !conversationId) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/messages`, {
        conversation_id: conversationId,
        message: message
      });
      setStats(res.data);
      await loadHistory();
      setMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    if (!conversationId) return;
    try {
      const res = await axios.get(`${API_URL}/conversations/${conversationId}/history`);
      setHistory(res.data.messages);
      setGoals(res.data.goals);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Conversational Agent Dashboard</h1>
      </header>
      
      <div className="container">
        {!conversationId ? (
          <div className="setup-panel">
            <h2>Start New Conversation</h2>
            <input
              type="text"
              placeholder="Enter User ID"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && createConversation()}
            />
            <button onClick={createConversation} disabled={!userId.trim()}>
              Create Conversation
            </button>
          </div>
        ) : (
          <>
            <div className="stats-panel">
              <div className="stat">
                <span className="label">State:</span>
                <span className="value">{stats.state || 'N/A'}</span>
              </div>
              <div className="stat">
                <span className="label">Messages:</span>
                <span className="value">{stats.total_messages || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Active Goals:</span>
                <span className="value">{stats.active_goals || 0}</span>
              </div>
              <div className="stat">
                <span className="label">Tokens:</span>
                <span className="value">{stats.total_tokens || 0}</span>
              </div>
            </div>

            <ConversationView history={history} goals={goals} />

            <div className="input-panel">
              <input
                type="text"
                placeholder="Type a message... (use /goal <text> to set a goal)"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !loading && sendMessage()}
                disabled={loading}
              />
              <button onClick={sendMessage} disabled={loading || !message.trim()}>
                {loading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
EOF

cat > frontend/src/App.css << 'EOF'
.App {
  min-height: 100vh;
  padding: 20px;
}

.App-header {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin-bottom: 20px;
}

.App-header h1 {
  color: #667eea;
  font-size: 28px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.setup-panel {
  background: white;
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  text-align: center;
}

.setup-panel h2 {
  margin-bottom: 20px;
  color: #333;
}

.setup-panel input {
  width: 100%;
  max-width: 400px;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 16px;
  margin-bottom: 15px;
}

.setup-panel button {
  padding: 12px 30px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}

.setup-panel button:hover:not(:disabled) {
  background: #5568d3;
}

.setup-panel button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.stats-panel {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.stat {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat .label {
  font-weight: 600;
  color: #666;
}

.stat .value {
  font-weight: 700;
  color: #667eea;
  font-size: 18px;
}

.input-panel {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  display: flex;
  gap: 10px;
}

.input-panel input {
  flex: 1;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 16px;
}

.input-panel button {
  padding: 12px 30px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}

.input-panel button:hover:not(:disabled) {
  background: #5568d3;
}

.input-panel button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
EOF

cat > frontend/src/components/ConversationView.js << 'EOF'
import React from 'react';
import './ConversationView.css';

function ConversationView({ history, goals }) {
  return (
    <div className="conversation-container">
      <div className="goals-panel">
        <h3>Active Goals</h3>
        {goals.length === 0 ? (
          <p className="empty">No goals set. Use /goal &lt;description&gt; to add one.</p>
        ) : (
          <div className="goals-list">
            {goals.map((goal, idx) => (
              <div key={idx} className={`goal ${goal.completed ? 'completed' : 'active'}`}>
                <div className="goal-icon">{goal.completed ? '✓' : '○'}</div>
                <div className="goal-content">
                  <div className="goal-description">{goal.description}</div>
                  <div className="goal-meta">
                    {new Date(goal.created_at).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="messages-panel">
        <h3>Conversation History</h3>
        {history.length === 0 ? (
          <p className="empty">No messages yet. Start the conversation!</p>
        ) : (
          <div className="messages-list">
            {history.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  <div className="message-meta">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                    {msg.token_count > 0 && ` • ${msg.token_count} tokens`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ConversationView;
EOF

cat > frontend/src/components/ConversationView.css << 'EOF'
.conversation-container {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 20px;
  margin-bottom: 20px;
}

.goals-panel, .messages-panel {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  padding: 20px;
}

.goals-panel h3, .messages-panel h3 {
  color: #333;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #f0f0f0;
}

.empty {
  color: #999;
  font-style: italic;
  text-align: center;
  padding: 20px;
}

.goals-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.goal {
  display: flex;
  gap: 10px;
  padding: 12px;
  border-radius: 8px;
  background: #f8f9fa;
}

.goal.completed {
  background: #d4edda;
}

.goal-icon {
  font-size: 20px;
  font-weight: bold;
}

.goal.completed .goal-icon {
  color: #28a745;
}

.goal.active .goal-icon {
  color: #667eea;
}

.goal-content {
  flex: 1;
}

.goal-description {
  font-weight: 500;
  color: #333;
  margin-bottom: 4px;
}

.goal-meta {
  font-size: 12px;
  color: #666;
}

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
  max-height: 500px;
  overflow-y: auto;
}

.message {
  display: flex;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
}

.message.user {
  background: #e3f2fd;
  align-self: flex-end;
}

.message.assistant {
  background: #f3e5f5;
}

.message-avatar {
  font-size: 24px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.message-content {
  flex: 1;
}

.message-text {
  color: #333;
  line-height: 1.5;
  margin-bottom: 6px;
}

.message-meta {
  font-size: 11px;
  color: #666;
}

@media (max-width: 768px) {
  .conversation-container {
    grid-template-columns: 1fr;
  }
}
EOF

# Helper scripts
cat > scripts/build.sh << 'EOF'
#!/bin/bash
echo "Building Conversational Agent..."

# Install backend dependencies
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
deactivate
cd ..

# Install frontend dependencies
cd frontend
if [ ! -d "node_modules" ]; then
    npm install --silent
fi
cd ..

echo "✓ Build complete"
EOF

cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "Starting Conversational Agent..."

# Start backend
cd backend
source venv/bin/activate
uvicorn api:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Start frontend
cd frontend
PORT=3000 npm start &
FRONTEND_PID=$!
cd ..

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
EOF

cat > scripts/test.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate

echo "Testing Conversational Agent..."
python3 - << 'PYTHON'
import asyncio
import sys
sys.path.insert(0, '.')
from conversation_engine import ConversationEngine
import os

async def test():
    api_key = os.getenv('GEMINI_API_KEY', '')
    engine = ConversationEngine(api_key, "test.db")
    await engine.initialize()
    
    # Test conversation creation
    conv_id = await engine.create_conversation("test_user")
    assert conv_id, "Failed to create conversation"
    print("✓ Conversation created")
    
    # Test message processing
    result = await engine.process_message(conv_id, "Hello!")
    assert "response" in result, "No response"
    print("✓ Message processed")
    
    # Test goal setting
    result = await engine.process_message(conv_id, "/goal Learn about AI")
    assert result["active_goals"] == 1, "Goal not set"
    print("✓ Goal setting works")
    
    # Test state persistence
    state = await engine.memory.load_state(conv_id)
    assert len(state.messages) == 4, "Messages not persisted"
    print("✓ State persistence works")
    
    print("\n✓ All tests passed!")
    
    # Cleanup
    import os
    if os.path.exists("test.db"):
        os.remove("test.db")

asyncio.run(test())
PYTHON

deactivate
EOF

cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "Stopping Conversational Agent..."
pkill -f "uvicorn api:app"
pkill -f "react-scripts"
echo "✓ Stopped"
EOF

chmod +x scripts/*.sh

# Tests
cat > tests/test_models.py << 'EOF'
import sys
sys.path.insert(0, '../backend')
from models import ConversationStateModel, Message, Goal
from datetime import datetime

def test_message_serialization():
    msg = Message(role="user", content="test")
    data = msg.dict()
    restored = Message.from_dict(data)
    assert restored.role == msg.role
    assert restored.content == msg.content
    print("✓ Message serialization")

def test_state_operations():
    state = ConversationStateModel(user_id="test", conversation_id="123")
    state.add_message("user", "hello", 10)
    state.add_goal("test goal")
    assert len(state.messages) == 1
    assert len(state.active_goals) == 1
    assert state.total_tokens == 10
    print("✓ State operations")

if __name__ == "__main__":
    test_message_serialization()
    test_state_operations()
    print("\n✓ All model tests passed")
EOF

# README
cat > README.md << 'EOF'
# L15: Basic Conversational Agent

Production-ready CLI conversational agent with persistent memory and goal-seeking behavior.

## Features
- Persistent conversation state across sessions
- Multi-user support with isolated conversations
- Goal tracking and completion detection
- Real-time monitoring dashboard
- SQLite persistence layer
- Gemini AI integration

## Quick Start

```bash
# Build
./scripts/build.sh

# Start (API + Dashboard)
./scripts/start.sh

# Or use CLI directly
cd backend
source venv/bin/activate
python cli.py

# Test
./scripts/test.sh
```

## Usage

### CLI Commands
- `/goal <description>` - Set a conversation goal
- `/quit` - Exit

### API Endpoints
- `POST /conversations` - Create conversation
- `POST /messages` - Send message
- `GET /conversations/{id}/history` - Get history

### Dashboard
Visit http://localhost:3000 for real-time monitoring

## Architecture
- **ConversationEngine**: Orchestrates message flow
- **MemoryManager**: SQLite persistence
- **GoalTracker**: LLM-based goal evaluation
- **GeminiClient**: API integration with retry logic

## Testing
```bash
cd backend
source venv/bin/activate
pytest tests/
```
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_NAME"
echo "2. ./scripts/build.sh"
echo "3. ./scripts/start.sh"
echo ""
echo "Dashboard: http://localhost:3000"
echo "API: http://localhost:8000"
echo ""

echo "✓ Setup script created: setup.sh"
echo "Run: bash setup.sh"