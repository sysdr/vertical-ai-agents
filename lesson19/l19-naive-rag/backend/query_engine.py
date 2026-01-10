"""Query engine orchestrating retrieve-then-generate workflow"""
import google.generativeai as genai
from typing import Dict, List
import time
import re

class QueryEngine:
    def __init__(self, knowledge_base, api_key: str):
        self.kb = knowledge_base
        genai.configure(api_key=api_key)
        
        # Try different model names in order of preference
        # Use model names without 'models/' prefix as that's how GenerativeModel expects them
        # Priority: Models with potentially better quota limits (lite versions first)
        self.model_candidates = [
            'gemini-2.0-flash-lite',  # Lite model - often has higher quota limits
            'gemini-2.0-flash-lite-001',  # Specific lite version
            'gemini-flash-lite-latest',  # Latest lite version
            'gemini-2.0-flash',      # Stable, fast model
            'gemini-2.5-flash-lite', # Latest stable lite model
            'gemini-2.5-flash',      # Latest stable flash model
            'gemini-pro-latest',     # Stable pro model (latest version)
            'gemini-2.0-flash-exp',  # Experimental but might have better quota
            'gemini-flash-latest',   # Generic latest flash
        ]
        self.models = []  # Store multiple models for fallback
        self.current_model_index = 0
        
        # Initialize multiple models for fallback capability
        for model_name in self.model_candidates:
            try:
                model = genai.GenerativeModel(model_name)
                self.models.append({'name': model_name, 'model': model})
                print(f"Successfully initialized model: {model_name}", file=__import__('sys').stderr)
            except Exception as e:
                print(f"Failed to initialize {model_name}: {str(e)}", file=__import__('sys').stderr)
                continue
        
        # If no candidates worked, try to discover available models dynamically
        if not self.models:
            try:
                print("Attempting to discover available models...", file=__import__('sys').stderr)
                available_models = genai.list_models()
                # Prioritize lite models for better quota
                lite_models = [m for m in available_models if 'lite' in m.name.lower() and 'generateContent' in m.supported_generation_methods]
                other_models = [m for m in available_models if 'lite' not in m.name.lower() and 'generateContent' in m.supported_generation_methods]
                
                for model_list in [lite_models, other_models]:
                    for model in model_list:
                        model_name = model.name.split('/')[-1]
                        try:
                            model_obj = genai.GenerativeModel(model_name)
                            self.models.append({'name': model_name, 'model': model_obj})
                            print(f"Successfully initialized discovered model: {model_name}", file=__import__('sys').stderr)
                        except:
                            continue
                    if self.models:
                        break
            except Exception as list_error:
                print(f"Error listing models: {str(list_error)}", file=__import__('sys').stderr)
        
        # Final fallback - raise error if no model could be initialized
        if not self.models:
            raise Exception("Could not initialize any Gemini model. Please check your API key and model availability.")
        
        print(f"Total models initialized for fallback: {len(self.models)}", file=__import__('sys').stderr)
        self.query_history = []
    
    def _get_current_model(self):
        """Get the current model being used"""
        if self.models:
            return self.models[self.current_model_index % len(self.models)]['model']
        return None
    
    def _switch_to_next_model(self):
        """Switch to the next available model (round-robin)"""
        if len(self.models) > 1:
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            model_name = self.models[self.current_model_index]['name']
            print(f"Switched to model: {model_name}", file=__import__('sys').stderr)
            return True
        return False
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Execute full RAG pipeline:
        1. Retrieve relevant chunks
        2. Build context
        3. Generate response
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        chunks = self.kb.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        if not chunks:
            return {
                'answer': "I don't have enough information to answer that question.",
                'chunks_used': [],
                'retrieval_time_ms': retrieval_time * 1000,
                'generation_time_ms': 0,
                'total_time_ms': retrieval_time * 1000
            }
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(chunks)
        ])
        
        # Generate response
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        generation_start = time.time()
        answer = None
        generation_time = 0
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                model = self._get_current_model()
                if model is None:
                    raise Exception("No model available")
                
                response = model.generate_content(prompt)
                generation_time = time.time() - generation_start
                
                # Handle response safely
                if hasattr(response, 'text') and response.text:
                    answer = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    # Try to extract text from candidates
                    answer = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else "No response generated"
                else:
                    answer = "Unable to generate a response from the model."
                
                # Success - break out of retry loop
                break
                
            except Exception as api_error:
                error_msg = str(api_error)
                
                # Check if it's a quota/rate limit error
                is_quota_error = "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
                
                if is_quota_error:
                    # Try to extract retry-after time from error message
                    retry_after_match = re.search(r'retry.*?(\d+(?:\.\d+)?)\s*(?:second|sec|s)', error_msg, re.IGNORECASE)
                    if retry_after_match:
                        retry_delay = float(retry_after_match.group(1)) + 0.5  # Add small buffer
                    else:
                        retry_delay = min(retry_delay * 2, 30)  # Exponential backoff, max 30 seconds
                    
                    # If we have more models to try, switch to next one
                    if attempt < max_retries - 1 and len(self.models) > 1 and self._switch_to_next_model():
                        print(f"Quota exceeded on model, trying next model. Retry {attempt + 1}/{max_retries}", file=__import__('sys').stderr)
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    elif attempt < max_retries - 1:
                        # No more models, wait and retry with same model
                        print(f"Quota exceeded. Waiting {retry_delay:.1f}s before retry {attempt + 1}/{max_retries}", file=__import__('sys').stderr)
                        time.sleep(retry_delay)
                        continue
                    else:
                        # All retries exhausted
                        generation_time = time.time() - generation_start
                        retry_after_msg = f" Please wait {int(retry_delay)} seconds before retrying." if retry_delay < 60 else " Please try again in a few minutes."
                        answer = f"API quota exceeded after {max_retries} attempts.{retry_after_msg} All available models have been tried. Check your API plan at https://ai.google.dev/gemini-api/docs/rate-limits"
                        break
                
                elif "401" in error_msg or "403" in error_msg or "API key" in error_msg.lower():
                    generation_time = time.time() - generation_start
                    answer = "API authentication failed. Please check your API key configuration. Error: " + error_msg[:200]
                    break  # Don't retry auth errors
                
                else:
                    # Other errors - try next model if available
                    if attempt < max_retries - 1 and len(self.models) > 1 and self._switch_to_next_model():
                        print(f"Error with model, trying next model. Retry {attempt + 1}/{max_retries}: {error_msg[:100]}", file=__import__('sys').stderr)
                        time.sleep(0.5)
                        continue
                    else:
                        generation_time = time.time() - generation_start
                        answer = f"Error generating response: {error_msg[:300]}"
                        break
        
        # If we exhausted all retries and still no answer
        if answer is None:
            generation_time = time.time() - generation_start
            answer = f"Failed to generate response after {max_retries} attempts. Please try again later."
        
        total_time = time.time() - start_time
        
        result = {
            'question': question,
            'answer': answer,
            'chunks_used': [
                {
                    'chunk_id': chunk['chunk_id'],
                    'text': chunk['text'][:200] + '...',
                    'score': chunk['score']
                }
                for chunk in chunks
            ],
            'retrieval_time_ms': round(retrieval_time * 1000, 2),
            'generation_time_ms': round(generation_time * 1000, 2),
            'total_time_ms': round(total_time * 1000, 2),
            'timestamp': time.time()
        }
        
        self.query_history.append(result)
        return result
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        return self.query_history[-limit:]
