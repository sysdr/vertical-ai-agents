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
