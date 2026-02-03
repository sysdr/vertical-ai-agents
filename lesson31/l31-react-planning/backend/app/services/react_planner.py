from app.models.schemas import (
    Observation, Reasoning, PlanningStep, PlanningTrace,
    PlanningRequest
)
from app.services.reasoning_engine import ReasoningEngine
from datetime import datetime
import uuid
from typing import List

class ReActPlanner:
    def __init__(self):
        self.reasoning_engine = ReasoningEngine()
        
    async def plan(self, request: PlanningRequest) -> PlanningTrace:
        """Execute ReAct planning loop"""
        task_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        # Initialize
        steps: List[PlanningStep] = []
        action_history: List[str] = []
        
        # Initial observation
        observation = Observation(
            iteration=0,
            task=request.task,
            context=request.initial_context
        )
        
        # Planning loop
        for i in range(request.max_iterations):
            observation.iteration = i + 1
            
            # Generate reasoning
            reasoning, metrics = await self.reasoning_engine.generate_reasoning(
                observation, action_history
            )
            
            # Create step
            step = PlanningStep(
                iteration=i + 1,
                observation=observation.model_copy(),
                reasoning=reasoning,
                metrics=metrics
            )
            steps.append(step)
            action_history.append(reasoning.action)
            
            # Check termination
            if reasoning.is_terminal:
                break
                
            # Update observation for next iteration
            observation = Observation(
                iteration=i + 2,
                task=request.task,
                previous_thought=reasoning.thought,
                previous_action=reasoning.action,
                context=self._update_context(observation.context, reasoning)
            )
        
        # Build trace
        completed_at = datetime.utcnow()
        total_latency = sum(step.metrics.get("latency_ms", 0) for step in steps)
        
        trace = PlanningTrace(
            task_id=task_id,
            task=request.task,
            steps=steps,
            status="COMPLETED" if steps[-1].reasoning.is_terminal else "MAX_ITERATIONS",
            total_iterations=len(steps),
            total_tokens=self.reasoning_engine.get_total_tokens(),
            total_latency_ms=total_latency,
            started_at=started_at,
            completed_at=completed_at
        )
        
        return trace
    
    def _update_context(self, current_context: dict, reasoning: Reasoning) -> dict:
        """Update context with new information"""
        updated = current_context.copy()
        updated["confidence_trend"] = updated.get("confidence_trend", []) + [reasoning.confidence]
        updated["action_count"] = updated.get("action_count", 0) + 1
        return updated
