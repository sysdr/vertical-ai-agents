"""
Reflexion Agent - Self-correcting ReAct agent with reflection loop
"""
from backend.react_agent import ReActAgent, Tool
from backend.reflection_engine import ReflectionEngine, ReflectionMemory
from typing import List, Dict, Any
import os
import re
import time
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
        self.max_reflections = max_reflections or int(os.getenv('MAX_REFLECTIONS', '4'))
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
            
            # On quota/rate-limit: wait and retry once before failing
            if self._is_quota_error(observation):
                retry_secs = self._parse_retry_seconds(observation)
                logger.warning(f"Quota exceeded, waiting {retry_secs}s before retry...")
                time.sleep(retry_secs)
                action, observation = self.step(task, reflection_history)
                if self._is_success(observation, task):
                    return {
                        'success': True,
                        'result': observation,
                        'attempts': attempts,
                        'reflections': reflection_history,
                        'session_id': self.current_session_id
                    }
                if self._is_quota_error(observation):
                    friendly_msg = (
                        "API quota exceeded (free tier: ~5 requests/min). "
                        "Wait ~1 min between batches of 4 queries, or upgrade at https://ai.google.dev/pricing"
                    )
                    logger.error(friendly_msg)
                    return {
                        'success': False,
                        'result': friendly_msg,
                        'attempts': attempts,
                        'reflections': reflection_history,
                        'session_id': self.current_session_id,
                        'error': 'QUOTA_EXCEEDED'
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
    
    def _is_quota_error(self, observation: str) -> bool:
        """Detect 429 quota/rate-limit errors"""
        obs_lower = observation.lower()
        return any(phrase in obs_lower for phrase in [
            '429',
            'exceeded your current quota',
            'quota exceeded',
            'rate limit',
            'please retry in'
        ])
    
    def _parse_retry_seconds(self, observation: str) -> float:
        """Extract retry delay from 429 error message"""
        match = re.search(r'retry in (\d+\.?\d*)s', observation, re.I)
        if match:
            return min(float(match.group(1)) + 1, 60)  # Add 1s buffer, cap at 60s
        return 6.0  # Default wait for free tier reset
    
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
