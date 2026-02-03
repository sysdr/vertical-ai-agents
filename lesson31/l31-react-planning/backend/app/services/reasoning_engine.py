import asyncio
import google.generativeai as genai
from app.core.config import settings
from app.models.schemas import Reasoning, Observation
import json
import re
import time
from typing import Dict, Any

class ReasoningEngine:
    def __init__(self):
        self._configured = False

    async def _generate_with_retry(self, prompt: str, max_retries: int = 2):
        """Call Gemini API with retry on 429 quota errors."""
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.to_thread(self.model.generate_content, prompt)
            except Exception as e:
                last_err = e
                err = str(e)
                if "429" in err and "retry" in err.lower() and attempt < max_retries:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err, re.I)
                    wait = min(float(match.group(1)) if match else 15, 30)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_err

    def _ensure_configured(self):
        if self._configured:
            return
        api_key = (settings.GEMINI_API_KEY or "").strip()
        if not api_key or api_key.startswith("your_") or "your_gemini" in api_key.lower():
            raise ValueError(
                "GEMINI_API_KEY not configured. Get a free key at https://aistudio.google.com/apikey "
                "and set it in backend/.env or l31-react-planning/.env. See API_KEY_SETUP.md"
            )
        genai.configure(api_key=api_key)
        self._configured = True
        # Use Flash-Lite for better free tier quota (15 RPM, 1000 RPD)
        safety = {cat: "BLOCK_ONLY_HIGH" for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]}
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite', safety_settings=safety)
        self.total_tokens = 0
        
    async def generate_reasoning(self, observation: Observation, action_history: list) -> tuple[Reasoning, Dict[str, Any]]:
        """Generate thought and action based on current observation"""
        self._ensure_configured()
        start_time = time.time()
        
        prompt = self._build_prompt(observation, action_history)
        
        try:
            response = await self._generate_with_retry(prompt)
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse structured response (handle blocked/empty)
            try:
                response_text = (response.text or "").strip()
            except (ValueError, AttributeError):
                response_text = ""
            if not response_text:
                raise ValueError("Empty response from model")
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            # Extract JSON object if wrapped in other text
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start >= 0 and end > start:
                response_text = response_text[start : end + 1]
            reasoning_data = json.loads(response_text)
            reasoning = Reasoning(**reasoning_data)
            
            # Collect metrics
            metrics = {
                "latency_ms": latency_ms,
                "tokens_used": len(prompt.split()) + len(response_text.split()),  # Approximation
                "model": "gemini-2.5-flash-lite",
                "timestamp": time.time()
            }
            
            self.total_tokens += metrics["tokens_used"]
            
            return reasoning, metrics
            
        except Exception as e:
            err = str(e)
            if "API key expired" in err or "API_KEY_INVALID" in err or "API key invalid" in err.lower():
                raise
            # Return user-friendly message for quota errors
            if "429" in err or "quota" in err.lower() or "exceeded" in err.lower():
                return Reasoning(
                    thought="Rate limit exceeded. Free tier has 15 requests/min. Wait a minute and try again, or check https://ai.google.dev/gemini-api/docs/rate-limits",
                    action="REQUEST_HUMAN_INTERVENTION",
                    confidence=0.0,
                    is_terminal=True
                ), {"latency_ms": (time.time() - start_time) * 1000, "error": "quota_exceeded"}
            return Reasoning(
                thought=f"Error in reasoning: {err}",
                action="REQUEST_HUMAN_INTERVENTION",
                confidence=0.0,
                is_terminal=True
            ), {"latency_ms": (time.time() - start_time) * 1000, "error": err}
    
    def _build_prompt(self, observation: Observation, action_history: list) -> str:
        """Build structured prompt for reasoning"""
        history_str = "\n".join([f"- {a}" for a in action_history[-5:]])  # Last 5 actions
        
        return f"""You are a ReAct planning agent. Given the current observation, reason about the next step.

CURRENT TASK: {observation.task}

CURRENT OBSERVATION:
Iteration: {observation.iteration}
Previous Thought: {observation.previous_thought or "N/A"}
Previous Action: {observation.previous_action or "N/A"}
Context: {json.dumps(observation.context, indent=2)}

ACTION HISTORY:
{history_str or "No previous actions"}

Instructions:
1. Analyze the current observation and context
2. Reason about what should happen next
3. Propose a specific action
4. Assess your confidence (0.0-1.0)
5. Determine if the task is complete

Respond with ONLY valid JSON (no markdown):
{{
  "thought": "Your detailed reasoning about the current state and next step",
  "action": "Specific next action to take",
  "confidence": 0.85,
  "is_terminal": false
}}

Set is_terminal to true if: task is complete, no more actions needed, or unrecoverable error.
"""

    def get_total_tokens(self) -> int:
        return self.total_tokens
