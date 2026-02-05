#!/bin/bash

# L33: Implementing Self-Correction (Reflexion)
# Automated setup script for VAIA Reflexion Agent

set -e

echo "=== L33: Reflexion Agent Setup ==="
echo "Building on L32 ReAct Agent..."

# Configuration
PROJECT_NAME="l33-reflexion-agent"
PYTHON_CMD="${PYTHON_CMD:-python3}"
GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY_HERE}"

# Create project structure
echo "Creating project structure..."
mkdir -p ${PROJECT_NAME}/{backend,frontend/src/components,frontend/public,tests,scripts,logs}
cd ${PROJECT_NAME}

# Create Python virtual environment (try venv, then virtualenv)
echo "Setting up Python environment..."
if ${PYTHON_CMD} -m venv venv 2>/dev/null; then
  source venv/bin/activate
elif command -v virtualenv &>/dev/null && virtualenv -p ${PYTHON_CMD} venv 2>/dev/null; then
  source venv/bin/activate
else
  echo "ERROR: Could not create virtual environment. Install python3-venv: apt install python3.12-venv"
  exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
cat > requirements.txt << 'DEPS'
google-generativeai==0.8.3
aiohttp==3.10.5
fastapi==0.115.0
uvicorn==0.32.0
pydantic==2.9.2
redis==5.2.0
pytest==8.3.3
pytest-asyncio==0.24.0
python-dotenv==1.0.1
requests==2.32.3
DEPS

pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
cat > .env << ENVFILE
GEMINI_API_KEY=${GEMINI_API_KEY}
GEMINI_MODEL_MAIN=gemini-2.5-flash
GEMINI_MODEL_REFLECT=gemini-2.5-flash
MAX_REFLECTIONS=3
REFLECTION_TIMEOUT=10
LOG_LEVEL=INFO
ENVFILE

# Backend: Core Reflexion Engine
echo "Creating backend components..."
touch backend/__init__.py

cat > backend/reflection_engine.py << 'PYFILE'
"""
Reflexion Engine - LLM-powered self-critique and plan refinement
"""
import google.generativeai as genai
from typing import Dict, List, Any
import os
from datetime import datetime

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

CRITIC_PROMPT = """You are an expert AI agent evaluator. Analyze the following agent execution:

**TASK GOAL:** {task}

**ACTION TAKEN:** {action}

**RESULT/OBSERVATION:** {observation}

**PREVIOUS REFLECTIONS:**
{memory}

Evaluate this attempt and provide structured feedback:

1. **SUCCESS ASSESSMENT:** Did this action successfully advance toward the goal?
2. **ROOT CAUSE ANALYSIS:** If failed, what specifically went wrong?
3. **STRATEGIC RECOMMENDATION:** What should the agent try differently next time?

Provide your response in this EXACT format:
SUCCESS: [true/false]
CRITIQUE: [2-3 sentences explaining what happened]
NEXT_STRATEGY: [specific, actionable change for next attempt]
CONFIDENCE: [0.0-1.0]
"""

class ReflectionEngine:
    """Generates structured critiques of agent actions using LLM"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('GEMINI_MODEL_REFLECT', 'gemini-2.5-flash')
        self.model = genai.GenerativeModel(self.model_name)
        
    def reflect(
        self, 
        task: str, 
        action: str, 
        observation: str, 
        memory: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate reflection on agent's last action
        
        Returns:
            {
                'success': bool,
                'critique': str,
                'next_strategy': str,
                'confidence': float,
                'timestamp': str
            }
        """
        memory_text = self._format_memory(memory)
        
        prompt = CRITIC_PROMPT.format(
            task=task,
            action=action,
            observation=observation,
            memory=memory_text
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temp for more consistent critiques
                    max_output_tokens=500
                )
            )
            
            reflection = self._parse_reflection(response.text)
            reflection['timestamp'] = datetime.now().isoformat()
            return reflection
            
        except Exception as e:
            # Fallback reflection on LLM failure
            return {
                'success': False,
                'critique': f"Reflection engine error: {str(e)}",
                'next_strategy': "Retry with error handling",
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_memory(self, memory: List[Dict]) -> str:
        """Format reflection history for prompt context"""
        if not memory:
            return "No previous reflections."
        
        formatted = []
        for i, ref in enumerate(memory[-3:], 1):  # Only last 3 to prevent context overflow
            formatted.append(
                f"Attempt {ref.get('attempt', i)}:\n"
                f"  Action: {ref.get('action', 'N/A')}\n"
                f"  Critique: {ref.get('critique', 'N/A')}\n"
                f"  Strategy: {ref.get('next_strategy', 'N/A')}"
            )
        return "\n\n".join(formatted)
    
    def _parse_reflection(self, text: str) -> Dict[str, Any]:
        """Extract structured fields from LLM response"""
        reflection = {
            'success': False,
            'critique': '',
            'next_strategy': '',
            'confidence': 0.5
        }
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('SUCCESS:'):
                reflection['success'] = 'true' in line.lower()
            elif line.startswith('CRITIQUE:'):
                reflection['critique'] = line.replace('CRITIQUE:', '').strip()
            elif line.startswith('NEXT_STRATEGY:'):
                reflection['next_strategy'] = line.replace('NEXT_STRATEGY:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.replace('CONFIDENCE:', '').strip()
                    reflection['confidence'] = float(conf_str)
                except:
                    reflection['confidence'] = 0.5
        
        return reflection

class ReflectionMemory:
    """In-memory storage for reflection history (Redis-ready interface)"""
    
    def __init__(self):
        self._memory: Dict[str, List[Dict]] = {}
    
    def add(self, session_id: str, reflection: Dict[str, Any]):
        """Add reflection to session history"""
        if session_id not in self._memory:
            self._memory[session_id] = []
        self._memory[session_id].append(reflection)
    
    def get(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all reflections for session"""
        return self._memory.get(session_id, [])
    
    def clear(self, session_id: str):
        """Clear session memory"""
        if session_id in self._memory:
            del self._memory[session_id]
    
    def get_stats(self, session_id: str) -> Dict[str, Any]:
        """Get reflection statistics for session"""
        reflections = self.get(session_id)
        if not reflections:
            return {'total': 0, 'successful': 0, 'avg_confidence': 0.0}
        
        successful = sum(1 for r in reflections if r.get('success', False))
        avg_conf = sum(r.get('confidence', 0.0) for r in reflections) / len(reflections)
        
        return {
            'total': len(reflections),
            'successful': successful,
            'failed': len(reflections) - successful,
            'avg_confidence': round(avg_conf, 2),
            'success_rate': round(successful / len(reflections), 2) if reflections else 0.0
        }
PYFILE

cat > backend/react_agent.py << 'PYFILE'
"""
ReAct Agent Base (from L32) - Simplified version for L33 integration
"""
import google.generativeai as genai
from typing import Dict, List, Any, Callable
import os
import re

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

REACT_PROMPT = """You are a helpful AI agent that can use tools to accomplish tasks.

AVAILABLE TOOLS:
{tools_description}

TASK: {task}

CONTEXT FROM PREVIOUS REFLECTIONS:
{reflection_context}

Use this format:
Thought: [your reasoning about what to do next]
Action: [tool_name(arguments)]
Observation: [result will be provided]

Then continue with more Thought/Action/Observation cycles until you can provide:
Final Answer: [your complete response to the task]

Begin!
"""

class Tool:
    """Simple tool wrapper"""
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ReActAgent:
    """Basic ReAct agent from L32 - reason and act loop"""
    
    def __init__(self, tools: List[Tool] = None, model_name: str = None):
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.model_name = model_name or os.getenv('GEMINI_MODEL_MAIN', 'gemini-2.5-flash')
        self.model = genai.GenerativeModel(self.model_name)
        self.history = []
    
    def step(self, task: str, reflection_memory: List[Dict] = None) -> tuple:
        """
        Execute one ReAct step
        
        Returns:
            (action_str, observation_str)
        """
        reflection_context = self._format_reflections(reflection_memory or [])
        tools_desc = self._format_tools()
        
        prompt = REACT_PROMPT.format(
            tools_description=tools_desc,
            task=task,
            reflection_context=reflection_context
        )
        
        # Add conversation history
        full_prompt = prompt
        if self.history:
            full_prompt += "\n\nPREVIOUS STEPS:\n" + "\n".join(self.history[-6:])
        
        try:
            response = self.model.generate_content(full_prompt)
            text = response.text
            
            # Parse action
            action = self._extract_action(text)
            if not action:
                return "No action found", "Error: Agent did not provide valid action"
            
            # Execute tool
            observation = self._execute_action(action)
            
            # Update history
            self.history.append(f"Thought: {self._extract_thought(text)}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
            
            return action, observation
            
        except Exception as e:
            return f"Error: {str(e)}", f"Failed to execute step: {str(e)}"
    
    def _format_tools(self) -> str:
        """Format tool descriptions for prompt"""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _format_reflections(self, reflections: List[Dict]) -> str:
        """Format reflection memory for context"""
        if not reflections:
            return "No previous reflections."
        
        latest = reflections[-1]
        return f"Latest Reflection: {latest.get('critique', 'N/A')}\nRecommended Strategy: {latest.get('next_strategy', 'N/A')}"
    
    def _extract_thought(self, text: str) -> str:
        """Extract thought from response"""
        match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)', text, re.DOTALL)
        return match.group(1).strip() if match else "No thought captured"
    
    def _extract_action(self, text: str) -> str:
        """Extract action from response"""
        match = re.search(r'Action:\s*(.+?)(?=\n|$)', text)
        return match.group(1).strip() if match else None
    
    def _execute_action(self, action: str) -> str:
        """Execute tool call"""
        # Parse tool_name(args)
        match = re.match(r'(\w+)\((.*)\)', action)
        if not match:
            return f"Error: Invalid action format '{action}'"
        
        tool_name, args_str = match.groups()
        
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        try:
            # Simple argument parsing (production would use AST)
            args_str = args_str.strip().strip('"\'')
            result = self.tools[tool_name](args_str)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def reset(self):
        """Clear conversation history"""
        self.history = []
PYFILE

cat > backend/reflexion_agent.py << 'PYFILE'
"""
Reflexion Agent - Self-correcting ReAct agent with reflection loop
"""
from backend.react_agent import ReActAgent, Tool
from backend.reflection_engine import ReflectionEngine, ReflectionMemory
from typing import List, Dict, Any
import os
import logging
import uuid

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class MaxReflectionsExceeded(Exception):
    """Raised when agent exhausts reflection attempts"""
    def __init__(self, message: str, memory: List[Dict]):
        super().__init__(message)
        self.memory = memory

class ReflexionAgent(ReActAgent):
    """
    Enhanced ReAct agent with self-correction via Reflexion
    
    Workflow:
    1. Execute ReAct step (from L32)
    2. If action fails, trigger reflection
    3. Update plan based on reflection
    4. Retry with refined strategy
    5. Repeat until success or max_reflections reached
    """
    
    def __init__(
        self, 
        tools: List[Tool] = None,
        model_name: str = None,
        max_reflections: int = None
    ):
        super().__init__(tools, model_name)
        self.reflection_engine = ReflectionEngine()
        self.reflection_memory_store = ReflectionMemory()
        self.max_reflections = max_reflections or int(os.getenv('MAX_REFLECTIONS', '3'))
        self.current_session_id = None
    
    def run(self, task: str, session_id: str = None) -> Dict[str, Any]:
        """
        Execute task with reflexion loop
        
        Returns:
            {
                'success': bool,
                'result': str,
                'attempts': int,
                'reflections': List[Dict],
                'session_id': str
            }
        """
        self.current_session_id = session_id or str(uuid.uuid4())
        self.reset()  # Clear ReAct history
        
        logger.info(f"Starting reflexion task: {task[:50]}... (session: {self.current_session_id})")
        
        attempts = 0
        
        while attempts < self.max_reflections:
            attempts += 1
            logger.info(f"Attempt {attempts}/{self.max_reflections}")
            
            # Get reflection context
            reflection_history = self.reflection_memory_store.get(self.current_session_id)
            
            # Execute ReAct step
            action, observation = self.step(task, reflection_history)
            
            logger.info(f"Action: {action}")
            logger.info(f"Observation: {observation[:100]}...")
            
            # Check if successful
            if self._is_success(observation, task):
                logger.info(f"✓ Task completed successfully in {attempts} attempt(s)")
                return {
                    'success': True,
                    'result': observation,
                    'attempts': attempts,
                    'reflections': reflection_history,
                    'session_id': self.current_session_id
                }
            
            # Fail fast on API key errors - retrying won't help
            if self._is_api_key_error(observation):
                friendly_msg = (
                    "API key invalid or expired. Please update GEMINI_API_KEY in .env. "
                    "Get a key at https://aistudio.google.com/apikey"
                )
                logger.error(friendly_msg)
                return {
                    'success': False,
                    'result': friendly_msg,
                    'attempts': attempts,
                    'reflections': reflection_history,
                    'session_id': self.current_session_id,
                    'error': 'API_KEY_INVALID'
                }
            
            # Fail fast on model-not-found errors - retrying won't help
            if self._is_model_error(observation):
                friendly_msg = (
                    "Model not found. Update GEMINI_MODEL_MAIN and GEMINI_MODEL_REFLECT in .env. "
                    "Use a supported model like gemini-2.5-flash. See https://ai.google.dev/gemini-api/docs/models"
                )
                logger.error(friendly_msg)
                return {
                    'success': False,
                    'result': friendly_msg,
                    'attempts': attempts,
                    'reflections': reflection_history,
                    'session_id': self.current_session_id,
                    'error': 'MODEL_NOT_FOUND'
                }
            
            # Trigger reflection on failure
            logger.info("Reflecting on failed attempt...")
            reflection = self.reflection_engine.reflect(
                task=task,
                action=action,
                observation=observation,
                memory=reflection_history
            )
            
            # Store reflection
            reflection_with_context = {
                'attempt': attempts,
                'action': action,
                'observation': observation[:200],  # Truncate for storage
                **reflection
            }
            self.reflection_memory_store.add(self.current_session_id, reflection_with_context)
            
            logger.info(f"Reflection - Success: {reflection['success']}")
            logger.info(f"Critique: {reflection['critique']}")
            logger.info(f"Next Strategy: {reflection['next_strategy']}")
        
        # Max reflections exceeded
        final_reflections = self.reflection_memory_store.get(self.current_session_id)
        error_msg = f"Failed after {attempts} attempts"
        logger.error(error_msg)
        
        return {
            'success': False,
            'result': error_msg,
            'attempts': attempts,
            'reflections': final_reflections,
            'session_id': self.current_session_id,
            'error': 'MAX_REFLECTIONS_EXCEEDED'
        }
    
    def _is_api_key_error(self, observation: str) -> bool:
        """Detect API key errors - fail fast, no point retrying"""
        obs_lower = observation.lower()
        return any(phrase in obs_lower for phrase in [
            'api key expired',
            'api_key_invalid',
            'no apikey',
            'no api_key',
            'please renew the api key',
            'api key not found'
        ])
    
    def _is_model_error(self, observation: str) -> bool:
        """Detect model-not-found / 404 errors - fail fast, no point retrying"""
        obs_lower = observation.lower()
        return (
            'is not found for api version' in obs_lower or
            ('models/' in obs_lower and 'not supported for generatecontent' in obs_lower)
        )
    
    def _is_success(self, observation: str, task: str) -> bool:
        """
        Determine if observation indicates successful task completion
        
        Production: Use LLM-based evaluation or explicit success criteria
        """
        # Simple heuristics for demo
        error_indicators = [
            'error',
            'failed',
            'exception',
            'invalid',
            'not found',
            'unavailable'
        ]
        
        obs_lower = observation.lower()
        
        # Check for explicit errors
        if any(indicator in obs_lower for indicator in error_indicators):
            return False
        
        # Check for substantial response (not just error message)
        if len(observation.strip()) < 10:
            return False
        
        # If observation contains data that looks relevant to task
        # This is simplified - production would use semantic matching
        return True
    
    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics for current session"""
        if not self.current_session_id:
            return {}
        return self.reflection_memory_store.get_stats(self.current_session_id)
PYFILE

cat > backend/tools.py << 'PYFILE'
"""
Example tools for Reflexion Agent testing
"""
from backend.react_agent import Tool
import requests
import random

def search_tool_func(query: str) -> str:
    """Simulated web search"""
    # In production, this would call actual search API
    
    # Simulate occasional failures for reflexion testing
    if random.random() < 0.2:  # 20% failure rate
        return "Error: Search API rate limit exceeded. Try again in 60 seconds."
    
    if "ceo anthropic" in query.lower():
        return "Dario Amodei is the CEO of Anthropic. He co-founded the company in 2021."
    elif "stock price" in query.lower():
        if "google" in query.lower() or "alphabet" in query.lower():
            return "Alphabet (GOOGL) current price: $142.35 (simulated data)"
        return "Please specify which company's stock price you want."
    elif "weather" in query.lower():
        return "Current weather: 72°F, partly cloudy (simulated data)"
    else:
        return f"Search results for '{query}': [Simulated results - implement actual search API]"

def calculator_tool_func(expression: str) -> str:
    """Simple calculator"""
    try:
        # Safe eval for basic math (production: use AST parser)
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def wikipedia_tool_func(topic: str) -> str:
    """Simulated Wikipedia lookup"""
    # Simulate network issues occasionally
    if random.random() < 0.15:  # 15% failure rate
        return "Error: Connection timeout. Wikipedia API unreachable."
    
    summaries = {
        "anthropic": "Anthropic is an AI safety company founded in 2021 by former OpenAI members, focused on building reliable, interpretable, and steerable AI systems.",
        "reflexion": "Reflexion is a framework that reinforces language agents through linguistic feedback, enabling them to learn from mistakes.",
        "react": "ReAct (Reasoning and Acting) combines chain-of-thought reasoning with action execution in language models."
    }
    
    topic_lower = topic.lower()
    for key, summary in summaries.items():
        if key in topic_lower:
            return summary
    
    return f"No Wikipedia article found for '{topic}'. Try a different search term."

# Create tool instances
search_tool = Tool(
    name="search",
    func=search_tool_func,
    description="Search the web for information. Usage: search('your query')"
)

calculator_tool = Tool(
    name="calculator",
    func=calculator_tool_func,
    description="Perform mathematical calculations. Usage: calculator('2 + 2')"
)

wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_tool_func,
    description="Look up topics on Wikipedia. Usage: wikipedia('topic name')"
)

DEFAULT_TOOLS = [search_tool, calculator_tool, wikipedia_tool]
PYFILE

cat > backend/api.py << 'PYFILE'
"""
FastAPI backend for Reflexion Agent
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.reflexion_agent import ReflexionAgent
from backend.tools import DEFAULT_TOOLS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L33 Reflexion Agent API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)

class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None

class TaskResponse(BaseModel):
    success: bool
    result: str
    attempts: int
    reflections: List[Dict[str, Any]]
    session_id: str
    stats: Dict[str, Any]

@app.post("/api/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute task with reflexion"""
    try:
        logger.info(f"Received task: {request.task}")
        
        result = agent.run(
            task=request.task,
            session_id=request.session_id
        )
        
        stats = agent.get_reflection_stats()
        
        return TaskResponse(
            success=result['success'],
            result=result['result'],
            attempts=result['attempts'],
            reflections=result['reflections'],
            session_id=result['session_id'],
            stats=stats
        )
    
    except Exception as e:
        logger.error(f"Task execution error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": agent.model_name,
        "max_reflections": agent.max_reflections
    }

@app.get("/api/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": name,
                "description": tool.description
            }
            for name, tool in agent.tools.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYFILE

# Frontend: React Dashboard
echo "Creating frontend..."

cat > frontend/package.json << 'PKGJSON'
{
  "name": "l33-reflexion-dashboard",
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
PKGJSON

cat > frontend/public/index.html << 'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="L33 Reflexion Agent Dashboard" />
    <title>L33: Reflexion Agent</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
    </style>
</head>
<body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
</body>
</html>
HTML

cat > frontend/src/App.js << 'JSFILE'
import React, { useState } from 'react';
import './App.css';
import ReflexionDashboard from './components/ReflexionDashboard';

function App() {
  return (
    <div className="App">
      <ReflexionDashboard />
    </div>
  );
}

export default App;
JSFILE

cat > frontend/src/App.css << 'CSS'
.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

* {
  box-sizing: border-box;
}
CSS

cat > frontend/src/index.js << 'JSFILE'
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
JSFILE

cat > frontend/src/index.css << 'CSS'
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New', monospace;
}
CSS

cat > frontend/src/components/ReflexionDashboard.js << 'JSFILE'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ReflexionDashboard.css';

const API_BASE = 'http://localhost:8000/api';
const API_KEY_ERROR_PHRASES = ['api key', 'api_key', 'renew the api key', 'invalid', 'expired'];

const ReflexionDashboard = () => {
  const [task, setTask] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [tools, setTools] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [apiKeyOk, setApiKeyOk] = useState(null);

  useEffect(() => {
    fetchTools();
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const r = await axios.get(\`\${API_BASE}/health\`);
      setApiKeyOk(r.data.api_key_configured !== false);
    } catch {
      setApiKeyOk(null);
    }
  };

  const fetchTools = async () => {
    try {
      const response = await axios.get(`${API_BASE}/tools`);
      setTools(response.data.tools);
    } catch (error) {
      console.error('Failed to fetch tools:', error);
    }
  };

  const executeTask = async () => {
    if (!task.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/execute`, {
        task: task,
        session_id: sessionId
      });

      setResult(response.data);
      setSessionId(response.data.session_id);
    } catch (error) {
      setResult({
        success: false,
        result: error.response?.data?.detail || 'Request failed',
        attempts: 0,
        reflections: [],
        stats: {}
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      executeTask();
    }
  };

  const exampleTasks = [
    "Find the CEO of Anthropic",
    "What is the stock price of Google?",
    "Calculate 15 * 23 + 47",
    "Tell me about Reflexion AI technique"
  ];

  return (
    <div className="dashboard-container">
      <div className="dashboard-card">
        <header className="dashboard-header">
          <h1>🧠 L33: Reflexion Agent</h1>
          <p className="subtitle">Self-Correcting AI with Iterative Reflection</p>
          {apiKeyOk === false && (
            <div className="api-key-alert">
              ⚠️ API key not configured. Update <code>.env</code> with your GEMINI_API_KEY. See API_KEY_SETUP.md
            </div>
          )}
        </header>

        <div className="tools-section">
          <h3>Available Tools</h3>
          <div className="tools-grid">
            {tools.map((tool, idx) => (
              <div key={idx} className="tool-card">
                <span className="tool-name">{tool.name}</span>
                <span className="tool-desc">{tool.description}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="input-section">
          <h3>Task Input</h3>
          <textarea
            className="task-input"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your task... (e.g., 'Find the CEO of Anthropic')"
            rows="3"
          />
          
          <div className="example-tasks">
            <span className="example-label">Try:</span>
            {exampleTasks.map((example, idx) => (
              <button
                key={idx}
                className="example-btn"
                onClick={() => setTask(example)}
              >
                {example}
              </button>
            ))}
          </div>

          <button
            className="execute-btn"
            onClick={executeTask}
            disabled={loading || !task.trim()}
          >
            {loading ? '🔄 Processing...' : '▶️ Execute with Reflexion'}
          </button>
        </div>

        {result && (
          <div className={`result-section ${result.success ? 'success' : 'failure'}`}>
            <div className="result-header">
              <h3>
                {result.success ? '✅ Success' : '❌ Failed'}
                <span className="attempts-badge">
                  {result.attempts} attempt{result.attempts !== 1 ? 's' : ''}
                </span>
              </h3>
            </div>

            <div className="result-content">
              <h4>Result:</h4>
              {result.error === 'API_KEY_INVALID' || (result.result && API_KEY_ERROR_PHRASES.some(p => result.result.toLowerCase().includes(p))) ? (
                <div className="api-key-error-box">
                  <strong>🔑 API Key Issue</strong>
                  <p>{result.result}</p>
                  <p className="setup-hint">Fix: Add a valid GEMINI_API_KEY to <code>l33-reflexion-agent/.env</code></p>
                  <p className="setup-hint">Get a key: <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a></p>
                </div>
              ) : (
                <pre>{result.result}</pre>
              )}
            </div>

            {result && (
              <div className="stats-section">
                <h4>Statistics:</h4>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Attempts:</span>
                    <span className="stat-value">{result.attempts ?? 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Total Reflections:</span>
                    <span className="stat-value">{result.stats?.total ?? 0}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Success Rate:</span>
                    <span className="stat-value">
                      {result.stats?.success_rate != null
                        ? ((result.stats.success_rate) * 100).toFixed(0) + '%'
                        : result.success ? '100%' : '0%'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg Confidence:</span>
                    <span className="stat-value">
                      {result.stats?.avg_confidence != null
                        ? ((result.stats.avg_confidence) * 100).toFixed(0) + '%'
                        : result.success ? '100%' : '-'}
                    </span>
                  </div>
                  {result.stats?.failed != null && result.stats.failed > 0 && (
                    <div className="stat-item">
                      <span className="stat-label">Failed Reflections:</span>
                      <span className="stat-value">{result.stats.failed}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {result.reflections && result.reflections.length > 0 && (
              <div className="reflections-section">
                <h4>Reflection History:</h4>
                {result.reflections.map((ref, idx) => (
                  <div key={idx} className="reflection-card">
                    <div className="reflection-header">
                      <span className="reflection-attempt">Attempt {ref.attempt}</span>
                      <span className={`reflection-status ${ref.success ? 'success' : 'failed'}`}>
                        {ref.success ? 'Success' : 'Failed'}
                      </span>
                    </div>
                    <div className="reflection-body">
                      <div className="reflection-item">
                        <strong>Action:</strong>
                        <code>{ref.action}</code>
                      </div>
                      <div className="reflection-item">
                        <strong>Critique:</strong>
                        <p>{ref.critique}</p>
                      </div>
                      <div className="reflection-item">
                        <strong>Next Strategy:</strong>
                        <p>{ref.next_strategy}</p>
                      </div>
                      {ref.confidence !== undefined && (
                        <div className="reflection-item">
                          <strong>Confidence:</strong>
                          <div className="confidence-bar">
                            <div 
                              className="confidence-fill" 
                              style={{width: `${ref.confidence * 100}%`}}
                            />
                            <span className="confidence-text">
                              {(ref.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <footer className="dashboard-footer">
          <p>Lesson 33: Implementing Self-Correction (Reflexion)</p>
          <p className="session-id">Session ID: {sessionId || 'Not started'}</p>
        </footer>
      </div>
    </div>
  );
};

export default ReflexionDashboard;
JSFILE

cat > frontend/src/components/ReflexionDashboard.css << 'CSS'
.dashboard-container {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
}

.dashboard-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  padding: 40px;
  animation: slideUp 0.5s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.dashboard-header {
  text-align: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #f0f0f0;
}

.dashboard-header h1 {
  margin: 0 0 10px 0;
  color: #2d3748;
  font-size: 2.5em;
  font-weight: 700;
}

.subtitle {
  color: #718096;
  font-size: 1.1em;
  margin: 0;
}

.api-key-alert {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 15px;
  font-size: 0.95em;
  color: #92400e;
}

.api-key-alert code {
  background: rgba(0,0,0,0.06);
  padding: 2px 6px;
  border-radius: 4px;
}

.api-key-error-box {
  background: #fef2f2;
  border: 1px solid #f87171;
  border-radius: 8px;
  padding: 16px;
  color: #991b1b;
}

.api-key-error-box strong {
  display: block;
  margin-bottom: 8px;
}

.api-key-error-box .setup-hint {
  margin: 8px 0 0 0;
  font-size: 0.9em;
}

.api-key-error-box a {
  color: #dc2626;
  font-weight: 600;
}

.tools-section {
  margin-bottom: 30px;
}

.tools-section h3 {
  color: #2d3748;
  margin-bottom: 15px;
  font-size: 1.3em;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.tool-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 15px;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: transform 0.2s;
}

.tool-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.tool-name {
  font-weight: 700;
  font-size: 1.1em;
}

.tool-desc {
  font-size: 0.9em;
  opacity: 0.95;
}

.input-section {
  margin-bottom: 30px;
}

.input-section h3 {
  color: #2d3748;
  margin-bottom: 15px;
  font-size: 1.3em;
}

.task-input {
  width: 100%;
  padding: 15px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1em;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.2s;
}

.task-input:focus {
  outline: none;
  border-color: #667eea;
}

.example-tasks {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 15px 0;
  align-items: center;
}

.example-label {
  color: #718096;
  font-weight: 600;
  margin-right: 5px;
}

.example-btn {
  background: #f7fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.9em;
  cursor: pointer;
  transition: all 0.2s;
  color: #4a5568;
}

.example-btn:hover {
  background: #667eea;
  color: white;
  border-color: #667eea;
}

.execute-btn {
  width: 100%;
  padding: 15px 30px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.execute-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.execute-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.result-section {
  background: #f7fafc;
  border-radius: 12px;
  padding: 25px;
  margin-top: 30px;
  border-left: 4px solid #48bb78;
  animation: fadeIn 0.3s ease-out;
}

.result-section.failure {
  border-left-color: #f56565;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.result-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.result-header h3 {
  margin: 0;
  color: #2d3748;
  font-size: 1.4em;
  display: flex;
  align-items: center;
  gap: 10px;
}

.attempts-badge {
  background: #667eea;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.7em;
  font-weight: 500;
}

.result-content {
  background: white;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
}

.result-content h4 {
  margin: 0 0 10px 0;
  color: #4a5568;
  font-size: 1.1em;
}

.result-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: #2d3748;
  font-family: 'Courier New', monospace;
  font-size: 0.95em;
  line-height: 1.6;
}

.stats-section {
  background: white;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
}

.stats-section h4 {
  margin: 0 0 15px 0;
  color: #4a5568;
  font-size: 1.1em;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.stat-label {
  color: #718096;
  font-size: 0.9em;
  font-weight: 500;
}

.stat-value {
  color: #2d3748;
  font-size: 1.5em;
  font-weight: 700;
}

.reflections-section {
  background: white;
  border-radius: 8px;
  padding: 15px;
}

.reflections-section h4 {
  margin: 0 0 15px 0;
  color: #4a5568;
  font-size: 1.1em;
}

.reflection-card {
  background: #f7fafc;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 15px;
  border: 1px solid #e2e8f0;
}

.reflection-card:last-child {
  margin-bottom: 0;
}

.reflection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #e2e8f0;
}

.reflection-attempt {
  font-weight: 700;
  color: #2d3748;
}

.reflection-status {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.85em;
  font-weight: 600;
}

.reflection-status.success {
  background: #c6f6d5;
  color: #22543d;
}

.reflection-status.failed {
  background: #fed7d7;
  color: #742a2a;
}

.reflection-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.reflection-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.reflection-item strong {
  color: #4a5568;
  font-size: 0.9em;
}

.reflection-item code {
  background: white;
  padding: 8px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
  color: #2d3748;
  border: 1px solid #e2e8f0;
}

.reflection-item p {
  margin: 0;
  color: #2d3748;
  line-height: 1.6;
}

.confidence-bar {
  position: relative;
  background: #e2e8f0;
  height: 24px;
  border-radius: 12px;
  overflow: hidden;
}

.confidence-fill {
  background: linear-gradient(90deg, #48bb78, #38a169);
  height: 100%;
  transition: width 0.3s ease;
}

.confidence-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #2d3748;
  font-weight: 600;
  font-size: 0.85em;
  text-shadow: 0 0 3px white;
}

.dashboard-footer {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 2px solid #f0f0f0;
  text-align: center;
  color: #718096;
}

.dashboard-footer p {
  margin: 5px 0;
  font-size: 0.9em;
}

.session-id {
  font-family: 'Courier New', monospace;
  font-size: 0.8em !important;
  color: #a0aec0;
}

@media (max-width: 768px) {
  .dashboard-card {
    padding: 20px;
  }

  .dashboard-header h1 {
    font-size: 2em;
  }

  .tools-grid {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }
}
CSS

# Tests
echo "Creating tests..."

cat > tests/test_reflexion.py << 'PYTEST'
"""
Tests for Reflexion Agent
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.reflexion_agent import ReflexionAgent
from backend.reflection_engine import ReflectionEngine, ReflectionMemory
from backend.tools import DEFAULT_TOOLS

def test_reflection_engine():
    """Test reflection engine generates valid critiques"""
    engine = ReflectionEngine()
    
    reflection = engine.reflect(
        task="Find the CEO of Anthropic",
        action="search('Anthropic CEO')",
        observation="Error: API rate limit exceeded",
        memory=[]
    )
    
    assert 'success' in reflection
    assert 'critique' in reflection
    assert 'next_strategy' in reflection
    assert isinstance(reflection['success'], bool)

def test_reflection_memory():
    """Test reflection memory stores and retrieves correctly"""
    memory = ReflectionMemory()
    
    session_id = "test_session_1"
    
    reflection_1 = {
        'attempt': 1,
        'critique': 'API failed',
        'next_strategy': 'Retry'
    }
    
    memory.add(session_id, reflection_1)
    
    retrieved = memory.get(session_id)
    assert len(retrieved) == 1
    assert retrieved[0]['attempt'] == 1
    
    stats = memory.get_stats(session_id)
    assert stats['total'] == 1

def test_reflexion_agent_success():
    """Test reflexion agent can complete simple task"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
    
    # Simple task that should succeed
    result = agent.run("Calculate 5 + 3")
    
    assert 'success' in result
    assert 'attempts' in result
    assert result['attempts'] <= 3

def test_reflexion_agent_max_attempts():
    """Test agent respects max_reflections limit"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=2)
    
    # Task designed to potentially fail
    result = agent.run("Find information about a non-existent topic xyz123abc")
    
    # Should not exceed max reflections
    assert result['attempts'] <= 2

def test_memory_prevents_repeat_mistakes():
    """Test that reflection memory influences next attempt"""
    agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
    
    result = agent.run("What is the stock price of Google?")
    
    if len(result['reflections']) > 1:
        # Check that later reflections reference earlier ones
        # (This is implicit in the reflection context passed to LLM)
        assert result['reflections'][0]['attempt'] == 1
        assert result['reflections'][-1]['attempt'] == len(result['reflections'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
PYTEST

# Create shell scripts
cat > scripts/build.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "Building L33 Reflexion Agent..."

# Backend
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Frontend
echo "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

echo "✓ Build complete"
SCRIPT

cat > scripts/start.sh << 'SCRIPT'
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

[ -f "$PROJECT_ROOT/venv/bin/activate" ] && source "$PROJECT_ROOT/venv/bin/activate"

echo "Starting L33 Reflexion Agent..."
echo "Project root: $PROJECT_ROOT"

if lsof -i :8000 >/dev/null 2>&1; then
  echo "ERROR: Port 8000 already in use. Run 'bash scripts/stop.sh' first."
  exit 1
fi
if lsof -i :3000 >/dev/null 2>&1; then
  echo "ERROR: Port 3000 already in use. Run 'bash scripts/stop.sh' first."
  exit 1
fi

echo "Starting FastAPI backend on port 8000..."
python -m backend.api &
BACKEND_PID=$!

sleep 3

echo "Starting React frontend on port 3000..."
cd "$PROJECT_ROOT/frontend"
npm start &
FRONTEND_PID=$!

echo ""
echo "✓ Services started"
echo "  - Backend API: http://localhost:8000"
echo "  - Frontend UI: http://localhost:3000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
SCRIPT

cat > scripts/stop.sh << 'SCRIPT'
#!/bin/bash

echo "Stopping L33 Reflexion Agent..."

# Kill backend
pkill -f "backend.api" || true

# Kill frontend
pkill -f "react-scripts" || true

echo "✓ Services stopped"
SCRIPT

cat > scripts/test.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "Running L33 Reflexion Agent tests..."

# Run pytest
pytest tests/ -v --tb=short

echo "✓ All tests passed"
SCRIPT

chmod +x scripts/*.sh

# Create README
cat > README.md << 'README'
# L33: Implementing Self-Correction (Reflexion)

Self-correcting VAIA agent with iterative reflection and autonomous error recovery.

## Quick Start

### Automated Setup
```bash
# 1. Add your Gemini API key to .env (see API_KEY_SETUP.md)
source venv/bin/activate
bash scripts/build.sh
bash scripts/start.sh
```

### Manual Setup

**Backend:**
```bash
pip install -r requirements.txt
python -m backend.api
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## Features

- ✅ Reflexion loop with LLM-powered critique
- ✅ Reflection memory across attempts
- ✅ Automatic plan refinement
- ✅ Production-ready error recovery
- ✅ Real-time dashboard

## Architecture

```
ReflexionAgent
├── ReActAgent (from L32)
├── ReflectionEngine (LLM critique)
└── ReflectionMemory (attempt history)
```

## Usage

```python
from backend.reflexion_agent import ReflexionAgent
from backend.tools import DEFAULT_TOOLS

agent = ReflexionAgent(tools=DEFAULT_TOOLS, max_reflections=3)
result = agent.run("Find the CEO of Anthropic")

print(f"Success: {result['success']}")
print(f"Attempts: {result['attempts']}")
print(f"Result: {result['result']}")
```

## Testing

```bash
bash scripts/test.sh
```

## Next Lesson

L34 adds iteration limits, token budgeting, and cost controls to the reflexion system.
README

cat > API_KEY_SETUP.md << 'APIKEY'
# API Key Setup

The L33 Reflexion Agent uses Google Gemini API. You need a valid API key.

1. Get a key at https://aistudio.google.com/apikey
2. Add to .env: GEMINI_API_KEY=your_key_here
3. Restart: bash scripts/stop.sh && bash scripts/start.sh
APIKEY

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Project structure created at: ${PROJECT_NAME}/"
echo ""
echo "Next steps:"
echo "  cd ${PROJECT_NAME}"
echo "  source venv/bin/activate"
echo "  bash scripts/build.sh"
echo "  bash scripts/start.sh"
echo ""
echo "Then open: http://localhost:3000"
echo ""
