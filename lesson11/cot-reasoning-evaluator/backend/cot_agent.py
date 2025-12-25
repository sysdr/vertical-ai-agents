"""Chain-of-Thought Agent with Reasoning Evaluation"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai

# Configure Gemini AI - use environment variable or fallback
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not set. API functionality will be limited.")

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
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            # Try models with better free tier support first
            # Start with flash-lite models which typically have better free tier quotas
            try:
                self.model = genai.GenerativeModel('models/gemini-flash-latest')
            except:
                try:
                    self.model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
                except:
                    try:
                        self.model = genai.GenerativeModel('models/gemini-pro-latest')
                    except:
                        # Last resort - use the 2.0 flash (may have quota limits)
                        self.model = genai.GenerativeModel('models/gemini-2.0-flash')
        else:
            self.model = None
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
        if self.model is None:
            raise ValueError("GEMINI_API_KEY is not set. Please set the environment variable to use this feature.")
        
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
