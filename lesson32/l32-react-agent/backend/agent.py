import google.generativeai as genai
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re
import logging
import time

logger = logging.getLogger(__name__)

# Retry config
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds
MAX_RATE_LIMIT_RETRIES = 5  # 429 quota exceeded

# Faster generation config: limit output tokens for ReAct steps (Thought/Action/Observation)
GENERATION_CONFIG = {
    "max_output_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.95,
}

class ReActAgent:
    """Production-grade ReAct agent with tool integration"""
    
    MODELS = ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']

    def __init__(self, tool_registry, api_key: str, max_iterations: int = 5):
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self._model_index = 0
        self.model = genai.GenerativeModel(
            self.MODELS[self._model_index],
            generation_config=GENERATION_CONFIG
        )
        self.conversation_history = []
        
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute ReAct loop for given query"""
        self.conversation_history = []
        reasoning_trace = []
        
        # Build tool descriptions for prompt
        tool_descriptions = self._build_tool_descriptions()
        
        # Initial prompt
        system_prompt = f"""You are a ReAct agent. Use tools to answer. Be concise.

Tools:
{tool_descriptions}

Format (use exactly):
Thought: [brief reasoning]
Action: [ToolName or FINISH]
Action Input: {{"param": "value"}}
Observation: [leave blank]

For final answer:
Thought: [summary]
Action: FINISH
Action Input: {{"answer": "Your answer"}}

Rules: Use exact tool names. One tool per step. Use FINISH when done."""

        current_query = f"{system_prompt}\n\nQuery: {query}\n\nRespond:"
        
        for iteration in range(self.max_iterations):
            try:
                # Generate response with retry for 504 Deadline Exceeded
                response_text = self._generate_with_retry(current_query)
                
                logger.info(f"Iteration {iteration + 1}: {response_text[:100]}...")
                
                # Parse response
                parsed = self._parse_response(response_text)
                
                if not parsed:
                    reasoning_trace.append({
                        'thought': 'Failed to parse LLM response',
                        'action': 'ERROR',
                        'action_input': {},
                        'observation': 'Invalid response format',
                        'timestamp': datetime.now().isoformat()
                    })
                    break
                
                thought = parsed['thought']
                action = parsed['action']
                action_input = parsed['action_input']
                
                # Check for completion
                if action == 'FINISH':
                    final_answer = action_input.get('answer', 'No answer provided')
                    reasoning_trace.append({
                        'thought': thought,
                        'action': action,
                        'action_input': action_input,
                        'observation': 'Task completed',
                        'timestamp': datetime.now().isoformat()
                    })
                    return {
                        'success': True,
                        'final_answer': final_answer,
                        'reasoning_trace': reasoning_trace,
                        'iterations': iteration + 1
                    }
                
                # Execute tool
                observation = self._execute_action(action, action_input)
                
                # Record step
                reasoning_trace.append({
                    'thought': thought,
                    'action': action,
                    'action_input': action_input,
                    'observation': observation,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update conversation for next iteration
                current_query += f"\n\n{response_text}\nObservation: {observation}\n\nContinue reasoning:"
                
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "exceeded" in err_str.lower()
                if is_rate_limit and iteration == 0:
                    fallback = self._direct_tool_fallback(query)
                    if fallback:
                        return fallback
                logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                reasoning_trace.append({
                    'thought': 'Execution error',
                    'action': 'ERROR',
                    'action_input': {},
                    'observation': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                break
        
        return {
            'success': False,
            'final_answer': 'Max iterations reached without completion',
            'reasoning_trace': reasoning_trace,
            'iterations': self.max_iterations
        }

    def _direct_tool_fallback(self, query: str) -> Optional[Dict[str, Any]]:
        """When API quota exhausted, answer using tools directly (no LLM) - unlimited demo mode"""
        q = query.lower().strip()
        reasoning_trace = []
        ts = datetime.now().isoformat()

        # Stock price: "google stock", "price of apple", "compare google and apple"
        stock_symbols = {'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
                         'apple': 'AAPL', 'aapl': 'AAPL', 'microsoft': 'MSFT', 'msft': 'MSFT',
                         'amazon': 'AMZN', 'amzn': 'AMZN', 'tesla': 'TSLA', 'tsla': 'TSLA'}
        if any(w in q for w in ['stock', 'price', 'share']):
            symbols_to_check = [sym for word, sym in stock_symbols.items() if word in q]
            if not symbols_to_check:
                symbols_to_check = ['GOOGL', 'AAPL']
            results = []
            for sym in symbols_to_check[:3]:
                r = self._execute_action('StockPrice', {'symbol': sym})
                if 'Tool error' not in r:
                    results.append(r)
            if results:
                answer = ' | '.join(results) + ' [Direct tool - no API quota used]'
                reasoning_trace.append({
                    'thought': 'API quota exhausted, using direct tool execution (unlimited demo mode)',
                    'action': 'StockPrice',
                    'action_input': {'symbols': symbols_to_check[:3]},
                    'observation': answer,
                    'timestamp': ts
                })
                return {'success': True, 'final_answer': answer, 'reasoning_trace': reasoning_trace, 'iterations': 1}

        # Calculator: "calculate", "what is X * Y", numbers with +-*/
        calc_match = re.search(r'(\d+(?:\s*[\+\-\*\/\%]\s*\d+)+|\d+\s*\*\s*\d+|\d+\s*\+\s*\d+)', query)
        if calc_match or 'calculate' in q or 'what is' in q and re.search(r'\d+.*[\+\-\*\/]', query):
            expr = calc_match.group(1).replace(' ', '') if calc_match else ''
            if not expr and 'calculate' in q:
                expr = re.sub(r'.*?calculate\s*', '', query, flags=re.I).replace(' ', '')
            if expr and re.match(r'^[\d\+\-\*\/\(\)\.\%]+$', expr):
                r = self._execute_action('Calculator', {'expression': expr})
                reasoning_trace.append({
                    'thought': 'API quota exhausted, using direct calculator',
                    'action': 'Calculator',
                    'action_input': {'expression': expr},
                    'observation': r,
                    'timestamp': ts
                })
                return {'success': True, 'final_answer': r + ' [Direct tool - no API quota used]', 'reasoning_trace': reasoning_trace, 'iterations': 1}

        # Wikipedia: "python", "tell me about", "what is X"
        wiki_terms = ['python', 'programming', 'wikipedia', 'tell me about', 'what is']
        if any(t in q for t in wiki_terms):
            if 'python' in q:
                term = 'Python (programming language)'
            elif 'about' in q:
                term = query.split('about')[-1].strip(' ?.,')
            elif 'what is' in q:
                term = query.split('what is')[-1].strip(' ?.,')
            else:
                term = query.strip(' ?.,')
            if len(term) > 2:
                r = self._execute_action('Wikipedia', {'query': term[:80]})
                if 'Tool error' not in r and 'not found' not in r.lower():
                    reasoning_trace.append({
                        'thought': 'API quota exhausted, using direct Wikipedia lookup',
                        'action': 'Wikipedia',
                        'action_input': {'query': term},
                        'observation': r,
                        'timestamp': ts
                    })
                    return {'success': True, 'final_answer': r + ' [Direct tool - no API quota used]', 'reasoning_trace': reasoning_trace, 'iterations': 1}

        # Weather
        if 'weather' in q:
            for loc in ['San Francisco', 'New York', 'London', 'Tokyo']:
                if loc.lower() in q:
                    r = self._execute_action('Weather', {'location': loc})
                    reasoning_trace.append({
                        'thought': 'API quota exhausted, using direct weather lookup',
                        'action': 'Weather',
                        'action_input': {'location': loc},
                        'observation': r,
                        'timestamp': ts
                    })
                    return {'success': True, 'final_answer': r + ' [Direct tool - no API quota used]', 'reasoning_trace': reasoning_trace, 'iterations': 1}
        return None
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text from response - handles multi-part responses (e.g. Gemini 2.5 thinking)"""
        try:
            return response.text
        except ValueError:
            pass
        try:
            parts = response.parts
        except (AttributeError, IndexError):
            parts = response.candidates[0].content.parts
        text_parts = []
        for part in parts:
            text = getattr(part, 'text', None) or (part.get('text') if isinstance(part, dict) else None)
            if text:
                text_parts.append(str(text))
        return '\n'.join(text_parts) if text_parts else ''

    def _parse_retry_delay(self, err_str: str) -> int:
        """Parse retry delay from 429 error - wait full suggested time + buffer"""
        m = re.search(r'retry\s+in\s+([\d.]+)\s*s', err_str, re.I)
        if m:
            return min(int(float(m.group(1))) + 5, 120)  # +5s buffer, max 2min
        m = re.search(r'retrydelay["\']?\s*:\s*["\']?(\d+)s?["\']?', err_str, re.I)
        if m:
            return min(int(m.group(1)) + 5, 120)
        return 35

    def _generate_with_retry(self, prompt: str) -> str:
        """Call Gemini with retry on 504 timeout and 429 rate limit"""
        last_error = None
        rate_limit_retries = 0
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                return self._extract_text_from_response(response)
            except Exception as e:
                last_error = e
                err_str = str(e)
                err_lower = err_str.lower()
                is_timeout = (
                    "504" in err_lower or "deadline" in err_lower or
                    "deadline_exceeded" in err_lower or "timeout" in err_lower
                )
                is_rate_limit = (
                    "429" in err_str or "quota" in err_lower or
                    "rate limit" in err_lower or "exceeded" in err_lower
                )
                is_daily_exhausted = "limit: 0" in err_str and ("perday" in err_lower or "per_day" in err_lower)
                if is_rate_limit and rate_limit_retries < MAX_RATE_LIMIT_RETRIES and not is_daily_exhausted:
                    if self._model_index < len(self.MODELS) - 1:
                        self._model_index += 1
                        self.model = genai.GenerativeModel(
                            self.MODELS[self._model_index],
                            generation_config=GENERATION_CONFIG
                        )
                        logger.warning(f"Rate limit on {self.MODELS[self._model_index-1]}, switching to {self.MODELS[self._model_index]}")
                        rate_limit_retries += 1
                        continue
                    delay = self._parse_retry_delay(err_str)
                    rate_limit_retries += 1
                    logger.warning(f"Rate limit (429), waiting {delay}s before retry {rate_limit_retries}/{MAX_RATE_LIMIT_RETRIES}...")
                    time.sleep(delay)
                    continue
                if is_timeout and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    logger.warning(f"API timeout (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        raise last_error

    def _build_tool_descriptions(self) -> str:
        """Build formatted tool descriptions for prompt"""
        descriptions = []
        for tool_name, tool in self.tool_registry.list_tools().items():
            schema = tool.schema
            descriptions.append(
                f"- {tool_name}: {schema['description']}\n"
                f"  Parameters: {json.dumps(schema['parameters'], indent=2)}"
            )
        return "\n".join(descriptions)
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured format"""
        try:
            # Extract Thought
            thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\Z)', response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # Extract Action
            action_match = re.search(r'Action:\s*(\w+)', response)
            action = action_match.group(1).strip() if action_match else None
            
            # Extract Action Input
            action_input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
            action_input = {}
            if action_input_match:
                action_input_str = action_input_match.group(1).strip()
                # Clean up potential formatting issues
                action_input_str = action_input_str.replace('\n', ' ')
                action_input = json.loads(action_input_str)
            
            if not action:
                return None
            
            return {
                'thought': thought,
                'action': action,
                'action_input': action_input
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None
    
    def _execute_action(self, action: str, args: Dict[str, Any]) -> str:
        """Execute tool action and return observation"""
        tool = self.tool_registry.get(action)
        if not tool:
            return f"Error: Tool '{action}' not found. Available tools: {list(self.tool_registry.list_tools().keys())}"
        
        try:
            result = tool.execute(**args)
            if result['success']:
                return result['result']
            else:
                return f"Tool error: {result['error']}"
        except Exception as e:
            return f"Execution error: {str(e)}"
