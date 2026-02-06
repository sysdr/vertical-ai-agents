from app.budget_manager import BudgetManager, BudgetException
from app.gemini_client import GeminiClient
from typing import Dict, List
import json

class BudgetedReflexionAgent:
    """ReAct agent with Reflexion and budget controls."""
    
    def __init__(self, llm_client: GeminiClient, budget_policy: Dict):
        self.llm = llm_client
        self.budget_manager = BudgetManager(budget_policy)
        self.trajectory = []
        
        # Simulated tools for demo
        self.tools = {
            'search': self._search_tool,
            'calculate': self._calculate_tool,
            'analyze': self._analyze_tool
        }
    
    async def run(self, task: str) -> str:
        """Execute task with budgeted ReAct and Reflexion loop."""
        try:
            while not self._is_task_complete():
                # ReAct planning step
                action = await self._plan_next_action(task)
                result = await self._execute_action(action)
                
                # Reflexion step (from L33)
                reflection = await self._reflect_on_step(action, result)
                
                self.trajectory.append({
                    'action': action,
                    'result': result,
                    'reflection': reflection
                })
                
                # Check if we should continue
                if self._should_terminate(reflection):
                    break
            
            return await self._synthesize_response()
        
        except BudgetException as e:
            # Graceful degradation
            return await self._handle_budget_violation(e)
    
    async def _plan_next_action(self, task: str) -> Dict:
        """Plan next action using ReAct reasoning."""
        context = self._format_trajectory()
        
        prompt = f"""Task: {task}

Previous steps:
{context}

Thought: What should I do next?
Reasoning: [Analyze the situation and determine the best next step]
Action: [Choose an action: search, calculate, or analyze]
Action Input: [Specific input for the action]

Respond in JSON format:
{{"thought": "...", "reasoning": "...", "action": "...", "action_input": "..."}}
"""
        
        response = await self.llm.generate_async(prompt)
        self.budget_manager.record_llm_call(
            response['input_tokens'],
            response['output_tokens'],
            'planning'
        )
        
        try:
            return json.loads(response['text'])
        except:
            return {
                'thought': 'Continue processing',
                'action': 'analyze',
                'action_input': task
            }
    
    async def _execute_action(self, action: Dict) -> str:
        """Execute the planned action."""
        action_name = action.get('action', 'analyze')
        action_input = action.get('action_input', '')
        
        if action_name in self.tools:
            return self.tools[action_name](action_input)
        return f"Executed {action_name} with input: {action_input}"
    
    async def _reflect_on_step(self, action: Dict, result: str) -> str:
        """Reflect on the action and result (Reflexion from L33)."""
        prompt = f"""Reflect on this step:
Action: {action.get('action')}
Result: {result}

Was this step helpful? What could be improved?
Should we continue or is the task complete?

Provide brief reflection and recommend: CONTINUE or COMPLETE
"""
        
        response = await self.llm.generate_async(prompt)
        self.budget_manager.record_llm_call(
            response['input_tokens'],
            response['output_tokens'],
            'reflection'
        )
        
        return response['text']
    
    def _search_tool(self, query: str) -> str:
        """Simulated search tool."""
        return f"Search results for '{query}': Found relevant information about the topic."
    
    def _calculate_tool(self, expression: str) -> str:
        """Simulated calculation tool."""
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except:
            return "Calculation failed"
    
    def _analyze_tool(self, text: str) -> str:
        """Simulated analysis tool."""
        return f"Analysis: The text contains {len(text)} characters and discusses important concepts."
    
    def _format_trajectory(self) -> str:
        """Format trajectory for context."""
        if not self.trajectory:
            return "No previous steps."
        
        formatted = []
        for i, step in enumerate(self.trajectory[-3:], 1):  # Last 3 steps
            formatted.append(f"Step {i}: {step['action'].get('action', 'unknown')}")
            formatted.append(f"Result: {step['result'][:100]}")
            formatted.append(f"Reflection: {step['reflection'][:100]}")
        
        return "\n".join(formatted)
    
    def _is_task_complete(self) -> bool:
        """Check if task is complete."""
        return len(self.trajectory) > 0 and any(
            'COMPLETE' in step['reflection'].upper()
            for step in self.trajectory[-2:]
        )
    
    def _should_terminate(self, reflection: str) -> bool:
        """Determine if we should terminate the loop."""
        return 'COMPLETE' in reflection.upper()
    
    async def _synthesize_response(self) -> str:
        """Synthesize final response from trajectory."""
        context = "\n\n".join([
            f"Step: {s['action'].get('action')}\nResult: {s['result']}"
            for s in self.trajectory
        ])
        
        prompt = f"""Based on these steps, provide a concise final answer:

{context}

Final Answer:"""
        
        response = await self.llm.generate_async(prompt)
        self.budget_manager.record_llm_call(
            response['input_tokens'],
            response['output_tokens'],
            'synthesis'
        )
        
        return response['text']
    
    async def _handle_budget_violation(self, exception: BudgetException) -> str:
        """Handle budget violations gracefully."""
        partial_result = "Task interrupted due to budget limits. "
        
        if self.trajectory:
            partial_result += f"Completed {len(self.trajectory)} steps. "
            partial_result += f"Last step: {self.trajectory[-1]['action'].get('action')}"
        
        return partial_result
    
    def get_metrics(self) -> Dict:
        """Get execution metrics."""
        metrics = self.budget_manager.get_usage_report()
        metrics['steps_completed'] = len(self.trajectory)
        return metrics
