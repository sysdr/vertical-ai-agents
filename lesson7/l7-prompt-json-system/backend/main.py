from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
import json
import re
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# Configure with explicit API version
try:
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Warning: Error configuring Gemini: {e}")
    genai.configure(api_key=api_key)

app = FastAPI(title="L7 Prompt Engineering System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class ParseStrategy(str, Enum):
    DIRECT = "direct"
    REGEX = "regex"
    REPAIR = "repair"
    FAILED = "failed"

class PromptRequest(BaseModel):
    instruction: str = Field(..., description="What you want the LLM to do")
    schema: Dict[str, Any] = Field(..., description="Expected JSON schema")
    temperature: float = Field(0.1, ge=0, le=2)

class ParseResult(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    strategy: ParseStrategy
    error: Optional[str] = None
    parse_time_ms: float
    attempts: int

class StructuredResponse(BaseModel):
    prompt_used: str
    raw_response: str
    parsed_data: Optional[Dict[str, Any]]
    parse_result: ParseResult
    validation_errors: List[str] = []
    timestamp: str

# Example schemas for testing
class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    interests: List[str]
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

class ProductInfo(BaseModel):
    product_name: str
    price: float
    category: str
    in_stock: bool
    rating: Optional[float] = None

# ============================================================================
# PROMPT ENGINE
# ============================================================================

class PromptEngine:
    @staticmethod
    def build_json_prompt(instruction: str, schema: dict, examples: Optional[List[dict]] = None) -> str:
        """Constructs a prompt that enforces JSON output"""
        
        prompt_parts = [
            instruction,
            "",
            "You must respond with ONLY valid JSON matching this exact schema:",
            json.dumps(schema, indent=2),
            "",
            "Critical rules:",
            "1. Return ONLY the JSON object",
            "2. No markdown code blocks (no ```json```)",
            "3. No explanatory text before or after",
            "4. All fields are required unless marked optional",
            "5. Use exact field names from schema",
        ]
        
        if examples:
            prompt_parts.extend([
                "",
                "Example valid responses:",
                *[json.dumps(ex, indent=2) for ex in examples]
            ])
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def build_repair_prompt(original_text: str, error: str) -> str:
        """Creates a prompt for LLM self-correction"""
        return f"""The previous response could not be parsed as JSON.

Original response:
{original_text}

Error encountered:
{error}

Please provide ONLY a corrected JSON response with no additional text."""

# ============================================================================
# JSON PARSER WITH FALLBACK STRATEGIES
# ============================================================================

class JSONParser:
    def __init__(self):
        # Initialize model with fallback strategy
        self.model = self._get_working_model()
    
    def _get_working_model(self):
        """Get a working Gemini model, trying multiple options"""
        # Try to get available models first
        try:
            available_models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name
                    # Extract short name (e.g., "models/gemini-pro" -> "gemini-pro")
                    if '/' in model_name:
                        short_name = model_name.split('/')[-1]
                    else:
                        short_name = model_name
                    available_models.append((model_name, short_name))
            
            if available_models:
                # Try the first available model's short name
                for full_name, short_name in available_models:
                    try:
                        return genai.GenerativeModel(short_name)
                    except Exception:
                        try:
                            return genai.GenerativeModel(full_name)
                        except Exception:
                            continue
        except Exception:
            pass
        
        # Fallback: try common model names (gemini-pro is most likely to work)
        model_names = ['gemini-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']
        for model_name in model_names:
            try:
                return genai.GenerativeModel(model_name)
            except Exception:
                continue
        
        # Last resort
        raise RuntimeError("Could not initialize any Gemini model. Please check your API key and model access.")
    
    async def parse_with_fallback(self, text: str, max_attempts: int = 3) -> ParseResult:
        """Multi-strategy JSON parsing with fallback"""
        start_time = datetime.now()
        
        # Strategy 1: Direct parse
        result = await self._try_direct_parse(text)
        if result.success:
            result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Strategy 2: Regex extraction
        result = await self._try_regex_extraction(text)
        if result.success:
            result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result
        
        # Strategy 3: LLM self-correction
        for attempt in range(max_attempts):
            result = await self._try_llm_repair(text, result.error)
            if result.success:
                result.attempts = attempt + 1
                result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                return result
            text = result.data if result.data else text
        
        result.parse_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result
    
    async def _try_direct_parse(self, text: str) -> ParseResult:
        """Strategy 1: Standard JSON parsing"""
        try:
            data = json.loads(text.strip())
            return ParseResult(
                success=True,
                data=data,
                strategy=ParseStrategy.DIRECT,
                parse_time_ms=0,
                attempts=1
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                strategy=ParseStrategy.DIRECT,
                error=str(e),
                parse_time_ms=0,
                attempts=1
            )
    
    async def _try_regex_extraction(self, text: str) -> ParseResult:
        """Strategy 2: Extract JSON from markdown or embedded text"""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',  # Markdown JSON block
            r'```\s*(\{.*?\})\s*```',       # Generic code block
            r'(\{[^{}]*\{.*?\}[^{}]*\})',   # Nested JSON
            r'(\{.*?\})',                    # Simple JSON
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return ParseResult(
                        success=True,
                        data=data,
                        strategy=ParseStrategy.REGEX,
                        parse_time_ms=0,
                        attempts=1
                    )
                except json.JSONDecodeError:
                    continue
        
        return ParseResult(
            success=False,
            strategy=ParseStrategy.REGEX,
            error="No valid JSON found in text",
            parse_time_ms=0,
            attempts=1
        )
    
    async def _try_llm_repair(self, text: str, error: str) -> ParseResult:
        """Strategy 3: Ask LLM to fix its own output"""
        try:
            repair_prompt = PromptEngine.build_repair_prompt(text, error)
            response = await asyncio.to_thread(
                self.model.generate_content,
                repair_prompt
            )
            
            repaired_text = response.text
            data = json.loads(repaired_text.strip())
            
            return ParseResult(
                success=True,
                data=data,
                strategy=ParseStrategy.REPAIR,
                parse_time_ms=0,
                attempts=1
            )
        except Exception as e:
            return ParseResult(
                success=False,
                data=repaired_text if 'repaired_text' in locals() else None,
                strategy=ParseStrategy.REPAIR,
                error=str(e),
                parse_time_ms=0,
                attempts=1
            )

# ============================================================================
# GEMINI CLIENT WITH STRUCTURED OUTPUT
# ============================================================================

class GeminiStructuredClient:
    def __init__(self):
        # Initialize model with fallback strategy
        self.model = self._get_working_model()
        self.parser = JSONParser()
    
    def _get_working_model(self):
        """Get a working Gemini model, trying multiple options"""
        # Try to get available models first
        try:
            available_models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name
                    # Extract short name (e.g., "models/gemini-pro" -> "gemini-pro")
                    if '/' in model_name:
                        short_name = model_name.split('/')[-1]
                    else:
                        short_name = model_name
                    available_models.append((model_name, short_name))
            
            if available_models:
                # Try the first available model's short name
                for full_name, short_name in available_models:
                    try:
                        return genai.GenerativeModel(short_name)
                    except Exception:
                        try:
                            return genai.GenerativeModel(full_name)
                        except Exception:
                            continue
        except Exception:
            pass
        
        # Fallback: try common model names (gemini-pro is most likely to work)
        model_names = ['gemini-pro', 'gemini-1.5-pro', 'gemini-1.5-flash']
        for model_name in model_names:
            try:
                return genai.GenerativeModel(model_name)
            except Exception:
                continue
        
        # Last resort
        raise RuntimeError("Could not initialize any Gemini model. Please check your API key and model access.")
    
    async def generate_structured(
        self,
        instruction: str,
        schema: dict,
        temperature: float = 0.1
    ) -> StructuredResponse:
        """Generate LLM response and parse to structured JSON"""
        
        # Build prompt
        prompt = PromptEngine.build_json_prompt(instruction, schema)
        
        try:
            # Call Gemini API with error handling and rate limit retry
            max_retries = 3
            retry_delay = 1  # Start with 1 second
            
            for attempt in range(max_retries):
                try:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                        )
                    )
                    raw_text = response.text
                    break  # Success, exit retry loop
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    error_lower = error_str.lower()
                    
                    # Handle rate limit / quota exceeded errors
                    if "429" in error_str or "quota" in error_lower or "rate limit" in error_lower:
                        # Extract retry delay from error if available
                        import re
                        retry_match = re.search(r'retry.*?(\d+)', error_str, re.IGNORECASE)
                        if retry_match:
                            retry_delay = int(retry_match.group(1)) + 1
                        else:
                            retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60s
                        
                        if attempt < max_retries - 1:
                            # Wait before retrying
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            # Last attempt failed
                            raise ValueError(
                                f"Rate limit exceeded: You've hit the free tier quota limit. "
                                f"Free tier allows 5 requests per minute per model. "
                                f"Please wait {retry_delay} seconds and try again, or upgrade your plan. "
                                f"See: https://ai.google.dev/gemini-api/docs/rate-limits"
                            )
                    
                    # Handle API key errors
                    if "403" in error_str or "api key" in error_lower or "leaked" in error_lower or "invalid" in error_lower:
                        raise ValueError(
                            "API key error: Your Gemini API key is invalid or has been disabled. "
                            "Please get a new API key from https://makersuite.google.com/app/apikey "
                            "and update it in the backend/.env file as GEMINI_API_KEY=your_new_key"
                        )
                    
                    # If model fails, try to reinitialize with a different model
                    if "404" in error_str or "not found" in error_lower or "not supported" in error_lower:
                        # Try to get a different working model
                        try:
                            self.model = self._get_working_model()
                            # Retry once with new model
                            response = await asyncio.to_thread(
                                self.model.generate_content,
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=temperature,
                                )
                            )
                            raw_text = response.text
                            break
                        except Exception:
                            if attempt < max_retries - 1:
                                continue
                            raise api_error
                    else:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        raise api_error
            
            # Parse with fallback strategies
            parse_result = await self.parser.parse_with_fallback(raw_text)
            
            # Validate against schema if parsing succeeded
            validation_errors = []
            if parse_result.success and parse_result.data:
                validation_errors = self._validate_schema(parse_result.data, schema)
            
            return StructuredResponse(
                prompt_used=prompt,
                raw_response=raw_text,
                parsed_data=parse_result.data,
                parse_result=parse_result,
                validation_errors=validation_errors,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return StructuredResponse(
                prompt_used=prompt,
                raw_response=str(e),
                parsed_data=None,
                parse_result=ParseResult(
                    success=False,
                    strategy=ParseStrategy.FAILED,
                    error=str(e),
                    parse_time_ms=0,
                    attempts=1
                ),
                validation_errors=[str(e)],
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_schema(self, data: dict, schema: dict) -> List[str]:
        """Basic schema validation"""
        errors = []
        
        # Check required fields
        required_fields = [k for k, v in schema.items() if not str(v).startswith("Optional")]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types (basic)
        for field, value in data.items():
            if field in schema:
                expected_type = schema[field]
                actual_type = type(value).__name__
                if expected_type != actual_type and not str(expected_type).startswith("Optional"):
                    errors.append(f"Field '{field}' has type {actual_type}, expected {expected_type}")
        
        return errors

# ============================================================================
# API ENDPOINTS
# ============================================================================

# Global client
client = GeminiStructuredClient()

# Metrics storage
metrics = {
    "total_requests": 0,
    "successful_parses": 0,
    "failed_parses": 0,
    "strategy_counts": {
        "direct": 0,
        "regex": 0,
        "repair": 0,
        "failed": 0
    },
    "avg_parse_time_ms": 0,
    "parse_times": []
}

@app.get("/")
async def root():
    return {
        "service": "L7 Prompt Engineering System",
        "status": "operational",
        "features": [
            "Smart prompt construction",
            "Multi-strategy JSON parsing",
            "Schema validation",
            "Real-time metrics"
        ]
    }

@app.post("/generate", response_model=StructuredResponse)
async def generate_structured_output(request: PromptRequest):
    """Generate structured JSON output from LLM"""
    
    result = await client.generate_structured(
        instruction=request.instruction,
        schema=request.schema,
        temperature=request.temperature
    )
    
    # Update metrics
    metrics["total_requests"] += 1
    if result.parse_result.success:
        metrics["successful_parses"] += 1
    else:
        metrics["failed_parses"] += 1
    
    metrics["strategy_counts"][result.parse_result.strategy] += 1
    metrics["parse_times"].append(result.parse_result.parse_time_ms)
    metrics["avg_parse_time_ms"] = sum(metrics["parse_times"]) / len(metrics["parse_times"])
    
    return result

@app.get("/metrics")
async def get_metrics():
    """Get parsing metrics"""
    return {
        **metrics,
        "success_rate": (
            metrics["successful_parses"] / metrics["total_requests"] * 100
            if metrics["total_requests"] > 0 else 0
        ),
        "failure_rate": (
            metrics["failed_parses"] / metrics["total_requests"] * 100
            if metrics["total_requests"] > 0 else 0
        )
    }

@app.post("/test/user-profile")
async def test_user_profile(name: str = "Alice Johnson"):
    """Test endpoint: Generate a user profile"""
    
    schema = {
        "name": "str",
        "age": "int",
        "email": "str",
        "interests": "List[str]"
    }
    
    instruction = f"Generate a realistic user profile for a person named {name}. Include their age, email, and 3-5 interests."
    
    return await client.generate_structured(instruction, schema)

@app.post("/test/product-info")
async def test_product_info(category: str = "electronics"):
    """Test endpoint: Generate product information"""
    
    schema = {
        "product_name": "str",
        "price": "float",
        "category": "str",
        "in_stock": "bool",
        "rating": "Optional[float]"
    }
    
    instruction = f"Generate information for a {category} product. Include realistic details."
    
    return await client.generate_structured(instruction, schema)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
