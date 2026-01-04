#!/bin/bash

# L16: Tool Use & Function Calling - Automated Setup
# Creates complete project with Gemini function calling integration

set -e

PROJECT_NAME="l16-tool-function-calling"
echo "🚀 Setting up L16: Tool Use & Function Calling"

# Create project structure
mkdir -p $PROJECT_NAME/{backend,frontend/src/components,frontend/public,docker,.venv}
cd $PROJECT_NAME

# ============================================
# Backend Setup
# ============================================

cat > backend/requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
google-generativeai==0.8.3
pydantic==2.9.2
python-dotenv==1.0.1
requests==2.32.3
python-multipart==0.0.12
aiofiles==24.1.0
EOF

cat > backend/.env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key_here
OPENWEATHER_API_KEY=demo
PORT=8000
EOF

cat > backend/main.py << 'EOF'
"""
L16: Tool Use & Function Calling
Production-grade function calling with Gemini AI
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI
app = FastAPI(title="L16 Tool Function Calling")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Tool Functions
# ============================================

def get_weather(city: str) -> Dict[str, Any]:
    """
    Fetch real weather data for a city.
    Uses OpenWeatherMap API with fallback to demo data.
    """
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY", "demo")
        
        # Demo mode for testing
        if api_key == "demo":
            logger.info(f"Demo mode: Simulating weather for {city}")
            demo_data = {
                "Tokyo": {"temp": 22, "condition": "Sunny", "humidity": 65},
                "London": {"temp": 15, "condition": "Cloudy", "humidity": 78},
                "New York": {"temp": 18, "condition": "Rainy", "humidity": 82},
                "Paris": {"temp": 20, "condition": "Clear", "humidity": 70},
            }
            data = demo_data.get(city, {"temp": 20, "condition": "Clear", "humidity": 60})
            return {
                "city": city,
                "temperature": data["temp"],
                "condition": data["condition"],
                "humidity": data["humidity"],
                "timestamp": datetime.now().isoformat(),
                "source": "demo"
            }
        
        # Real API call
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": city,
            "temperature": round(data["main"]["temp"], 1),
            "condition": data["weather"][0]["main"],
            "humidity": data["main"]["humidity"],
            "timestamp": datetime.now().isoformat(),
            "source": "openweathermap"
        }
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching weather for {city}")
        return {"error": "Weather service timeout", "city": city}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather for {city}: {e}")
        return {"error": f"Failed to fetch weather: {str(e)}", "city": city}
    except Exception as e:
        logger.error(f"Unexpected error in get_weather: {e}")
        return {"error": f"Unexpected error: {str(e)}", "city": city}


# Tool registry mapping function names to implementations
TOOL_FUNCTIONS = {
    "get_weather": get_weather
}

# Tool definitions for Gemini
TOOL_DECLARATIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a specific city including temperature, conditions, and humidity",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city to get weather for (e.g., 'Tokyo', 'London', 'New York')"
                }
            },
            "required": ["city"]
        }
    }
]

# ============================================
# Tool Execution
# ============================================

def execute_tool(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool function with given arguments.
    Includes error handling and logging.
    """
    logger.info(f"Executing tool: {function_name} with args: {arguments}")
    
    if function_name not in TOOL_FUNCTIONS:
        error_msg = f"Unknown function: {function_name}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    try:
        func = TOOL_FUNCTIONS[function_name]
        result = func(**arguments)
        logger.info(f"Tool {function_name} completed successfully")
        return result
    except TypeError as e:
        error_msg = f"Invalid arguments for {function_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

# ============================================
# Models
# ============================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []
    conversation_id: str
    timestamp: str

# ============================================
# Conversation State
# ============================================

# In-memory conversation storage (extends L15 pattern)
conversations: Dict[str, List[Dict[str, Any]]] = {}

def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve conversation history for given ID"""
    return conversations.get(conversation_id, [])

def save_conversation_turn(conversation_id: str, user_msg: str, assistant_msg: str, tool_calls: List[Dict] = None):
    """Save conversation turn to history"""
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        "user": user_msg,
        "assistant": assistant_msg,
        "tool_calls": tool_calls or [],
        "timestamp": datetime.now().isoformat()
    })

# ============================================
# Agent Orchestration
# ============================================

def chat_with_tools(user_message: str, conversation_id: str) -> tuple[str, List[Dict[str, Any]]]:
    """
    Main orchestration function implementing multi-turn tool calling.
    
    Flow:
    1. Send user message to Gemini with tool definitions
    2. Check if Gemini wants to call tools
    3. If yes: execute tools, send results back to Gemini
    4. Get final natural language response
    5. Return response and tool call log
    """
    
    try:
        # Initialize Gemini model with tools
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp',
            tools=TOOL_DECLARATIONS
        )
        
        # Get conversation history
        history = get_conversation_history(conversation_id)
        
        # Build context from history
        context_messages = []
        for turn in history[-5:]:  # Last 5 turns for context
            context_messages.append(f"User: {turn['user']}")
            context_messages.append(f"Assistant: {turn['assistant']}")
        
        # Construct prompt with context
        if context_messages:
            full_prompt = "\n".join(context_messages) + f"\nUser: {user_message}"
        else:
            full_prompt = user_message
        
        logger.info(f"Sending to Gemini: {user_message}")
        
        # First Gemini call - might return tool calls
        response = model.generate_content(full_prompt)
        
        tool_calls_made = []
        
        # Check if Gemini wants to use tools
        if response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Check for function call
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = dict(function_call.args)
                    
                    logger.info(f"Tool call requested: {function_name}({function_args})")
                    
                    # Execute the tool
                    tool_result = execute_tool(function_name, function_args)
                    
                    # Log the tool call
                    tool_calls_made.append({
                        "function": function_name,
                        "arguments": function_args,
                        "result": tool_result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Second Gemini call with tool result
                    result_message = f"Tool '{function_name}' returned: {json.dumps(tool_result)}"
                    logger.info(f"Sending tool result back to Gemini")
                    
                    # Create new conversation with tool result
                    final_response = model.generate_content([
                        full_prompt,
                        f"\n\nTool result: {result_message}\n\nPlease provide a natural language response based on this data."
                    ])
                    
                    final_text = final_response.text
                    return final_text, tool_calls_made
        
        # No tool calls needed, return direct response
        return response.text, tool_calls_made
        
    except Exception as e:
        logger.error(f"Error in chat_with_tools: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {
        "service": "L16 Tool Function Calling",
        "status": "running",
        "tools_available": len(TOOL_DECLARATIONS),
        "tools": [tool["name"] for tool in TOOL_DECLARATIONS]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with tool calling support.
    Implements multi-turn conversation with function execution.
    """
    
    conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"
    
    try:
        # Get response with tool execution
        response_text, tool_calls = chat_with_tools(request.message, conversation_id)
        
        # Save to conversation history
        save_conversation_turn(conversation_id, request.message, response_text, tool_calls)
        
        return ChatResponse(
            response=response_text,
            tool_calls=tool_calls,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Retrieve conversation history"""
    history = get_conversation_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "turns": len(history),
        "history": history
    }

@app.get("/tools")
async def list_tools():
    """List available tools and their schemas"""
    return {
        "tools": TOOL_DECLARATIONS,
        "count": len(TOOL_DECLARATIONS)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF

# ============================================
# Frontend Setup
# ============================================

cat > frontend/package.json << 'EOF'
{
  "name": "l16-tool-calling-ui",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1",
    "axios": "^1.7.7"
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
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="L16 Tool Function Calling Demo" />
    <title>L16: Tool Function Calling</title>
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
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
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
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New', monospace;
}
EOF

cat > frontend/src/App.js << 'EOF'
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ChatInterface from './components/ChatInterface';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [toolCalls, setToolCalls] = useState([]);
  const [stats, setStats] = useState({ totalTools: 0, totalMessages: 0 });

  // Load tools on mount
  useEffect(() => {
    loadTools();
  }, []);

  const loadTools = async () => {
    try {
      const response = await axios.get(`${API_URL}/tools`);
      setStats(prev => ({ ...prev, totalTools: response.data.count }));
    } catch (error) {
      console.error('Error loading tools:', error);
    }
  };

  const sendMessage = async (messageText) => {
    if (!messageText.trim()) return;

    // Add user message
    const userMessage = { role: 'user', content: messageText, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        message: messageText,
        conversation_id: conversationId
      });

      // Update conversation ID
      if (!conversationId) {
        setConversationId(response.data.conversation_id);
      }

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(response.data.timestamp),
        toolCalls: response.data.tool_calls
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Update tool calls history
      if (response.data.tool_calls.length > 0) {
        setToolCalls(prev => [...prev, ...response.data.tool_calls]);
      }

      // Update stats
      setStats(prev => ({
        ...prev,
        totalMessages: prev.totalMessages + 1
      }));

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'error',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setConversationId(null);
    setToolCalls([]);
    setStats(prev => ({ ...prev, totalMessages: 0 }));
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🛠️ L16: Tool Use & Function Calling</h1>
        <p>Production-grade agent with Gemini function calling</p>
      </header>

      <div className="main-container">
        <div className="stats-bar">
          <div className="stat">
            <span className="stat-label">Tools Available</span>
            <span className="stat-value">{stats.totalTools}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Messages</span>
            <span className="stat-value">{stats.totalMessages}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Tool Calls</span>
            <span className="stat-value">{toolCalls.length}</span>
          </div>
        </div>

        <ChatInterface
          messages={messages}
          onSendMessage={sendMessage}
          isLoading={isLoading}
          onClear={clearChat}
        />

        {toolCalls.length > 0 && (
          <div className="tool-calls-panel">
            <h3>Recent Tool Calls</h3>
            {toolCalls.slice(-5).reverse().map((call, idx) => (
              <div key={idx} className="tool-call-item">
                <div className="tool-call-header">
                  <span className="tool-name">{call.function}</span>
                  <span className="tool-time">
                    {new Date(call.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="tool-call-args">
                  Args: {JSON.stringify(call.arguments)}
                </div>
                <div className="tool-call-result">
                  {call.result.error ? (
                    <span className="error-badge">Error: {call.result.error}</span>
                  ) : (
                    <span className="success-badge">✓ Success</span>
                  )}
                </div>
              </div>
            ))}
          </div>
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
  display: flex;
  flex-direction: column;
}

.app-header {
  background: rgba(255, 255, 255, 0.95);
  padding: 2rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.app-header h1 {
  color: #667eea;
  margin-bottom: 0.5rem;
  font-size: 2rem;
}

.app-header p {
  color: #666;
  font-size: 1rem;
}

.main-container {
  flex: 1;
  max-width: 1400px;
  width: 100%;
  margin: 2rem auto;
  padding: 0 1rem;
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 1rem;
}

.stats-bar {
  grid-column: 1 / -1;
  display: flex;
  gap: 1rem;
  background: rgba(255, 255, 255, 0.95);
  padding: 1rem;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.stat {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0.5rem;
}

.stat-label {
  font-size: 0.875rem;
  color: #666;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #667eea;
}

.tool-calls-panel {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 10px;
  padding: 1rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  max-height: 600px;
  overflow-y: auto;
}

.tool-calls-panel h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1rem;
}

.tool-call-item {
  background: #f8f9fa;
  border-left: 3px solid #667eea;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
  border-radius: 5px;
  font-size: 0.875rem;
}

.tool-call-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.tool-name {
  font-weight: bold;
  color: #667eea;
}

.tool-time {
  color: #999;
  font-size: 0.75rem;
}

.tool-call-args {
  color: #666;
  margin-bottom: 0.5rem;
  word-break: break-all;
}

.tool-call-result {
  margin-top: 0.5rem;
}

.success-badge {
  background: #4caf50;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
}

.error-badge {
  background: #f44336;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
}

@media (max-width: 1024px) {
  .main-container {
    grid-template-columns: 1fr;
  }
  
  .tool-calls-panel {
    max-height: 300px;
  }
}
EOF

cat > frontend/src/components/ChatInterface.js << 'EOF'
import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

function ChatInterface({ messages, onSendMessage, isLoading, onClear }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const quickQuestions = [
    "What's the weather in Tokyo?",
    "How's the weather in London and Paris?",
    "Tell me about the weather in New York"
  ];

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Chat with Tool-Augmented Agent</h2>
        <button onClick={onClear} className="clear-btn">Clear Chat</button>
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>👋 Welcome to L16: Function Calling</h3>
            <p>Ask about weather in any city to see function calling in action!</p>
            <div className="quick-questions">
              <p>Try these:</p>
              {quickQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => !isLoading && onSendMessage(q)}
                  className="quick-question-btn"
                  disabled={isLoading}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-header">
              <span className="message-role">
                {msg.role === 'user' ? '👤 You' : msg.role === 'assistant' ? '🤖 Agent' : '⚠️ Error'}
              </span>
              <span className="message-time">
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
            <div className="message-content">{msg.content}</div>
            {msg.toolCalls && msg.toolCalls.length > 0 && (
              <div className="message-tools">
                <span className="tools-badge">
                  🛠️ {msg.toolCalls.length} tool call{msg.toolCalls.length > 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="message assistant loading">
            <div className="message-header">
              <span className="message-role">🤖 Agent</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about weather in any city..."
          disabled={isLoading}
          className="message-input"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="send-btn"
        >
          {isLoading ? '⏳' : '→'}
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;
EOF

cat > frontend/src/components/ChatInterface.css << 'EOF'
.chat-interface {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  height: 600px;
  overflow: hidden;
}

.chat-header {
  padding: 1rem;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-header h2 {
  font-size: 1.25rem;
  color: #333;
  margin: 0;
}

.clear-btn {
  background: #f44336;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background 0.2s;
}

.clear-btn:hover {
  background: #d32f2f;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.welcome-message {
  text-align: center;
  padding: 2rem;
  color: #666;
}

.welcome-message h3 {
  color: #667eea;
  margin-bottom: 0.5rem;
}

.quick-questions {
  margin-top: 1.5rem;
}

.quick-questions p {
  margin-bottom: 0.75rem;
  font-weight: 500;
}

.quick-question-btn {
  display: block;
  width: 100%;
  max-width: 400px;
  margin: 0.5rem auto;
  padding: 0.75rem;
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 5px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  color: #333;
}

.quick-question-btn:hover:not(:disabled) {
  background: #667eea;
  color: white;
  border-color: #667eea;
}

.quick-question-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.message {
  padding: 1rem;
  border-radius: 10px;
  max-width: 80%;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message.user {
  align-self: flex-end;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.message.assistant {
  align-self: flex-start;
  background: #f8f9fa;
  color: #333;
  border: 1px solid #e0e0e0;
}

.message.error {
  align-self: center;
  background: #ffebee;
  color: #c62828;
  border: 1px solid #ef5350;
}

.message.loading {
  align-self: flex-start;
}

.message-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
  opacity: 0.8;
}

.message-role {
  font-weight: 600;
}

.message-time {
  font-size: 0.75rem;
}

.message-content {
  line-height: 1.6;
}

.message-tools {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.tools-badge {
  background: #667eea;
  color: white;
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-size: 0.75rem;
  font-weight: 500;
}

.typing-indicator {
  display: flex;
  gap: 0.25rem;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #667eea;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.typing-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.typing-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

.input-form {
  padding: 1rem;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 0.5rem;
}

.message-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}

.message-input:focus {
  border-color: #667eea;
}

.message-input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.send-btn {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1.25rem;
  transition: transform 0.2s;
}

.send-btn:hover:not(:disabled) {
  transform: scale(1.05);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
EOF

# ============================================
# Docker Configuration
# ============================================

cat > docker/Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["python", "main.py"]
EOF

cat > docker/Dockerfile.frontend << 'EOF'
FROM node:20-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .

EXPOSE 3000

CMD ["npm", "start"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY:-demo}
    volumes:
      - ./backend:/app
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend/src:/app/src
    depends_on:
      - backend
    restart: unless-stopped
EOF

# ============================================
# Helper Scripts
# ============================================

cat > build.sh << 'EOF'
#!/bin/bash
echo "🔨 Building L16 Tool Function Calling..."

# Check for Docker
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
    deactivate
    cd ..
    
    # Frontend
    cd frontend
    npm install
    cd ..
fi

echo "✅ Build complete!"
EOF

cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting L16 Tool Function Calling..."

if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    echo "Starting with Docker..."
    docker-compose up -d
    echo "✅ Services started!"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
else
    echo "Starting without Docker..."
    
    # Start backend
    cd backend
    source venv/bin/activate
    python main.py &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    BROWSER=none npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo "✅ Services started!"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
fi
EOF

cat > stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping L16 Tool Function Calling..."

if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    docker-compose down
else
    pkill -f "python main.py"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped!"
EOF

cat > test.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing L16 Tool Function Calling..."

# Wait for backend
echo "Waiting for backend..."
sleep 5

# Test health endpoint
echo -e "\n1. Testing health endpoint..."
curl -s http://localhost:8000/health | python3 -m json.tool

# Test tools list
echo -e "\n2. Testing tools endpoint..."
curl -s http://localhost:8000/tools | python3 -m json.tool

# Test weather function
echo -e "\n3. Testing weather function call..."
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?"}' | python3 -m json.tool

echo -e "\n4. Testing multi-city query..."
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Compare weather in London and Paris"}' | python3 -m json.tool

echo -e "\n✅ Tests complete!"
EOF

chmod +x build.sh start.sh stop.sh test.sh

# ============================================
# Documentation
# ============================================

cat > README.md << 'EOF'
# L16: Tool Use & Function Calling

Production-grade function calling system with Gemini AI.

## Features

- ✅ Weather function with real API integration
- ✅ Multi-turn conversation loops
- ✅ Tool call logging and monitoring
- ✅ Error handling and fallbacks
- ✅ Extensible tool registry

## Quick Start

### With Docker
```bash
./build.sh
./start.sh
```

### Without Docker
```bash
./build.sh
./start.sh
```

Access:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Testing

```bash
./test.sh
```

## Architecture

- **Backend**: FastAPI with Gemini function calling
- **Frontend**: React with real-time chat
- **Tools**: Weather API with error handling
- **State**: In-memory conversation tracking

## Tool Functions

### get_weather(city: str)
Fetches current weather for a city.

Demo mode returns simulated data for:
- Tokyo, London, New York, Paris

## API Endpoints

- `POST /chat` - Send message, execute tools
- `GET /history/{id}` - Get conversation history
- `GET /tools` - List available tools
- `GET /health` - Health check

## Next Steps (L17)

- Add Pydantic schema validation
- Implement tool versioning
- Add complex nested schemas
- Build tool catalog system
EOF

echo "✅ Setup complete! Project structure created in $PROJECT_NAME/"
echo ""
echo "Next steps:"
echo "  cd $PROJECT_NAME"
echo "  ./build.sh    # Build dependencies"
echo "  ./start.sh    # Start services"
echo "  ./test.sh     # Run tests"
echo ""
echo "Access the application:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"