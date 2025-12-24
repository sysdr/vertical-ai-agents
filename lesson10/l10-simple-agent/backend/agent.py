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
        self.api_key = api_key
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
        
        # Try different model names - will be determined on first use
        self.model_name = None
        self.model = None
    
    def _get_model(self):
        """Initialize model with fallback to available models"""
        if self.model is not None:
            return self.model
        
        # Try different model names in order of preference
        model_names = ['gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
        
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                # Test if model works by making a simple call
                test_response = self.model.generate_content("test")
                return self.model
            except Exception:
                continue
        
        # If all fail, try to list available models
        try:
            models = genai.list_models()
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.split('/')[-1]
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        self.model_name = model_name
                        return self.model
                    except:
                        continue
        except:
            pass
        
        # Last resort: use gemini-pro anyway (might work with different API version)
        self.model = genai.GenerativeModel('gemini-pro')
        self.model_name = 'gemini-pro'
        return self.model
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
            model = self._get_model()
            response = await asyncio.to_thread(
                model.generate_content,
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
            model = self._get_model()
            response = await asyncio.to_thread(
                model.generate_content,
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
