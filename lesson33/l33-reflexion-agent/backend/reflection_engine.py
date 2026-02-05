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
