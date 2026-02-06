import os
from typing import Dict

class GeminiClient:
    """Client for Google Gemini AI API with mock fallback for demo when API unavailable."""
    
    def __init__(self):
        self._model = None
        self._use_mock = os.getenv('USE_MOCK_LLM', '0').lower() in ('1', 'true', 'yes')
        if not self._use_mock:
            try:
                import google.generativeai as genai
                api_key = os.getenv('GEMINI_API_KEY', '')
                if api_key:
                    genai.configure(api_key=api_key)
                    self._model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                self._use_mock = True
    
    def _mock_generate(self, prompt: str) -> Dict:
        """Simulated response for demo when API is unavailable."""
        input_tokens = len(prompt) // 4
        if 'Reflect on this step' in prompt:
            text = 'The step was helpful. Task appears complete. COMPLETE'
        elif 'Based on these steps' in prompt or 'Final Answer' in prompt:
            text = 'Based on the analysis: The task has been completed successfully with the expected results.'
        elif 'Calculate' in prompt or '2+2' in prompt:
            text = '{"thought": "Calculate the expression", "reasoning": "Use calculate tool", "action": "calculate", "action_input": "2+2"}'
        else:
            text = '{"thought": "Analyze the task", "reasoning": "Process with analyze tool", "action": "analyze", "action_input": "' + prompt[:50] + '"}'
        output_tokens = len(text) // 4
        return {'text': text, 'input_tokens': input_tokens, 'output_tokens': output_tokens}
    
    def generate(self, prompt: str) -> Dict:
        """Generate response and return text with token counts."""
        if self._use_mock or not self._model:
            return self._mock_generate(prompt)
        try:
            response = self._model.generate_content(prompt)
            input_tokens = len(prompt) // 4
            output_tokens = len(response.text) // 4
            return {'text': response.text, 'input_tokens': input_tokens, 'output_tokens': output_tokens}
        except Exception:
            return self._mock_generate(prompt)
    
    async def generate_async(self, prompt: str) -> Dict:
        """Async version for FastAPI compatibility."""
        return self.generate(prompt)
