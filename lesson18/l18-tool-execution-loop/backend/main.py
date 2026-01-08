from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable
import google.generativeai as genai
import json
import time
import asyncio
import traceback
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found!")
    print("   The server will start, but chat functionality will not work.")
    print("   To fix: Create backend/.env with: GEMINI_API_KEY=your_api_key_here")
    print("   Get API key from: https://makersuite.google.com/app/apikey")
    GEMINI_API_KEY = "NOT_SET"  # Placeholder to allow server to start
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini model (default to a generally-available model; experimental models may have free-tier limit=0)
GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL")
    or os.getenv("GOOGLE_GEMINI_MODEL")
    or "gemini-2.0-flash"
)

app = FastAPI(title="L18 Tool Execution Loop")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# TOOL SCHEMAS (from L17)
# ============================================================================

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Optional[Callable] = Field(default=None, exclude=True)

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def calculate_revenue(year: int, quarter: int) -> Dict[str, Any]:
    """Calculate revenue for given period"""
    time.sleep(0.1)  # Simulate processing
    base = 1000000
    growth = 0.15
    quarters_since_2020 = (year - 2020) * 4 + quarter
    revenue = base * ((1 + growth) ** quarters_since_2020)
    return {
        "year": year,
        "quarter": quarter,
        "revenue_usd": round(revenue, 2),
        "growth_rate": growth,
        "currency": "USD"
    }

def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get weather information for a city"""
    time.sleep(0.2)  # Simulate API call
    temps = {"new york": 22, "london": 18, "tokyo": 25, "sydney": 28, "paris": 20}
    temp_c = temps.get(city.lower(), 20)
    temp_f = (temp_c * 9/5) + 32 if unit == "fahrenheit" else temp_c
    return {
        "city": city,
        "temperature": round(temp_f if unit == "fahrenheit" else temp_c, 1),
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 12
    }

def search_products(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search product catalog"""
    time.sleep(0.15)
    products = [
        {"id": 1, "name": "Laptop Pro", "price": 1299, "rating": 4.5},
        {"id": 2, "name": "Wireless Mouse", "price": 29, "rating": 4.2},
        {"id": 3, "name": "USB-C Hub", "price": 49, "rating": 4.7},
        {"id": 4, "name": "Monitor 27\"", "price": 399, "rating": 4.6},
        {"id": 5, "name": "Keyboard Mechanical", "price": 159, "rating": 4.8},
    ]
    results = [p for p in products if query.lower() in p["name"].lower()][:max_results]
    return {
        "query": query,
        "results": results,
        "total_found": len(results),
        "search_time_ms": 150
    }

def document_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search knowledge base (RAG preview for L19)"""
    time.sleep(0.3)
    docs = [
        {"id": "doc1", "title": "VAIA Architecture", "relevance": 0.95},
        {"id": "doc2", "title": "Tool Execution Patterns", "relevance": 0.87},
        {"id": "doc3", "title": "Multi-Agent Systems", "relevance": 0.82},
    ]
    return {
        "query": query,
        "documents": docs[:top_k],
        "total_searched": 1000,
        "search_method": "vector_similarity"
    }

# ============================================================================
# FUNCTION REGISTRY
# ============================================================================

class FunctionRegistry:
    def __init__(self):
        self._functions: Dict[str, Dict[str, Any]] = {}
        self._execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_latency_ms": 0,
            "by_function": {}
        }
    
    def register(self, schema: ToolSchema, function: Callable):
        """Register a function with its schema"""
        self._functions[schema.name] = {
            "schema": schema,
            "function": function,
            "call_count": 0,
            "success_count": 0,
            "total_latency": 0
        }
        print(f"✓ Registered tool: {schema.name}")
    
    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get schema for a function"""
        if name in self._functions:
            return self._functions[name]["schema"]
        return None
    
    def execute(self, name: str, inputs: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Execute a function with safety wrapper"""
        if name not in self._functions:
            return {"error": f"Unknown function: {name}", "success": False}
        
        func_info = self._functions[name]
        func = func_info["function"]
        
        start_time = time.time()
        self._execution_stats["total_calls"] += 1
        func_info["call_count"] += 1
        
        try:
            # Execute with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, **inputs)
                try:
                    result = future.result(timeout=timeout)
                    
                    # Success metrics
                    latency = (time.time() - start_time) * 1000
                    self._execution_stats["successful_calls"] += 1
                    func_info["success_count"] += 1
                    func_info["total_latency"] += latency
                    
                    return {
                        "success": True,
                        "result": result,
                        "latency_ms": round(latency, 2),
                        "function": name
                    }
                except FuturesTimeoutError:
                    self._execution_stats["failed_calls"] += 1
                    return {
                        "success": False,
                        "error": f"Execution timeout after {timeout}s",
                        "function": name
                    }
        except Exception as e:
            self._execution_stats["failed_calls"] += 1
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "function": name
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        by_function = {}
        for name, info in self._functions.items():
            avg_latency = (info["total_latency"] / info["call_count"]) if info["call_count"] > 0 else 0
            success_rate = (info["success_count"] / info["call_count"]) if info["call_count"] > 0 else 0
            by_function[name] = {
                "calls": info["call_count"],
                "successes": info["success_count"],
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 2)
            }
        
        return {
            **self._execution_stats,
            "by_function": by_function
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM"""
        schemas = []
        for name, info in self._functions.items():
            schema = info["schema"]
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description
                        }
                        for param in schema.parameters
                    },
                    "required": [p.name for p in schema.parameters if p.required]
                }
            })
        return schemas

# Initialize registry
registry = FunctionRegistry()

# Register tools
registry.register(
    ToolSchema(
        name="calculate_revenue",
        description="Calculate company revenue for a specific year and quarter",
        parameters=[
            ToolParameter(name="year", type="integer", description="Year (e.g., 2024)"),
            ToolParameter(name="quarter", type="integer", description="Quarter (1-4)")
        ]
    ),
    calculate_revenue
)

registry.register(
    ToolSchema(
        name="get_weather",
        description="Get current weather information for a city",
        parameters=[
            ToolParameter(name="city", type="string", description="City name"),
            ToolParameter(name="unit", type="string", description="Temperature unit (celsius or fahrenheit)", required=False)
        ]
    ),
    get_weather
)

registry.register(
    ToolSchema(
        name="search_products",
        description="Search product catalog by keyword",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query"),
            ToolParameter(name="max_results", type="integer", description="Maximum results to return", required=False)
        ]
    ),
    search_products
)

registry.register(
    ToolSchema(
        name="document_search",
        description="Search knowledge base documents (RAG)",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query"),
            ToolParameter(name="top_k", type="integer", description="Number of results", required=False)
        ]
    ),
    document_search
)

# ============================================================================
# TOOL EXECUTOR
# ============================================================================

class ExecutionState:
    READY = "ready"
    CALLING_LLM = "calling_llm"
    PARSING_RESPONSE = "parsing_response"
    EXECUTING_TOOLS = "executing_tools"
    BUILDING_CONTEXT = "building_context"
    FALLBACK = "fallback"
    COMPLETE = "complete"
    FAILED = "failed"

class ToolExecutor:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.api_key_set = GEMINI_API_KEY != "NOT_SET"
        self.api_key_valid = False
        self.api_key_error = None
        self.model_name = GEMINI_MODEL
        self.fallback_enabled = os.getenv("ALLOW_FALLBACK_PLANNER", "true").strip().lower() in (
            "1", "true", "yes", "y", "on"
        )
        
        if self.api_key_set:
            try:
                # Try to initialize the model to validate the API key
                self.model = genai.GenerativeModel(
                    self.model_name,
                    tools=self._build_tool_declarations()
                )
                # Test with a simple call to verify the key works
                try:
                    test_response = self.model.generate_content("test")
                    self.api_key_valid = True
                    print("✅ API key validated successfully")
                except Exception as e:
                    error_str = str(e).lower()
                    model_not_found = (
                        "is not found for api version" in error_str
                        or ("models/" in error_str and "not found" in error_str)
                        or ("not supported for generatecontent" in error_str)
                    )
                    # Check for quota/rate limit errors (429)
                    if "429" in str(e) or "quota" in error_str or "rate limit" in error_str:
                        self.api_key_valid = False
                        err_text = str(e)
                        if "limit: 0" in err_text and "free_tier" in err_text:
                            self.api_key_error = f"QUOTA_ZERO: {err_text}"
                        else:
                            self.api_key_error = f"QUOTA_EXCEEDED: {err_text}"
                        self.model = None
                        if "limit: 0" in err_text and "free_tier" in err_text:
                            print("⚠️  This project has 0 free-tier quota for generateContent (plan/billing issue).")
                            print("   Fix options:")
                            print("   1. Create a NEW API key in a NEW project (AI Studio) and try again")
                            print("   2. Enable billing / upgrade plan for this project")
                            print("   3. Check usage/quota: https://ai.dev/usage?tab=rate-limit")
                        else:
                            print("⚠️  API quota exceeded (rate limit)")
                            print("   Solutions:")
                            print("   1. Wait and retry (rate limits reset)")
                            print("   2. Get a new API key from https://makersuite.google.com/app/apikey")
                            print("   3. Upgrade to paid tier for higher limits")
                            print("   Meanwhile, use /execute-tool endpoint for direct tool testing")
                    elif "api" in error_str and ("key" in error_str or "invalid" in error_str or "expired" in error_str):
                        self.api_key_valid = False
                        self.api_key_error = str(e)
                        self.model = None
                        print(f"⚠️  API key validation failed: {str(e)[:100]}")
                        print("   Run: ./scripts/fix_api_key.sh to fix this")
                    elif model_not_found:
                        self.api_key_valid = False
                        self.api_key_error = f"MODEL_NOT_FOUND: {str(e)}"
                        self.model = None
                        print("⚠️  Model name not supported/available for this API key/library.")
                        print(f"   Current model: {self.model_name}")
                        print("   Fix: set GEMINI_MODEL=gemini-2.0-flash and restart backend")
                    else:
                        # Other error, but key might be valid
                        self.api_key_valid = True
            except Exception as e:
                error_str = str(e).lower()
                model_not_found = (
                    "is not found for api version" in error_str
                    or ("models/" in error_str and "not found" in error_str)
                    or ("not supported for generatecontent" in error_str)
                )
                # Check for quota/rate limit errors
                if "429" in str(e) or "quota" in error_str or "rate limit" in error_str:
                    self.api_key_valid = False
                    err_text = str(e)
                    if "limit: 0" in err_text and "free_tier" in err_text:
                        self.api_key_error = f"QUOTA_ZERO: {err_text}"
                        print("⚠️  This project has 0 free-tier quota for generateContent (plan/billing issue).")
                        print("   Fix options:")
                        print("   1. Create a NEW API key in a NEW project (AI Studio) and try again")
                        print("   2. Enable billing / upgrade plan for this project")
                        print("   3. Check usage/quota: https://ai.dev/usage?tab=rate-limit")
                    else:
                        self.api_key_error = f"QUOTA_EXCEEDED: {err_text}"
                        print("⚠️  API quota exceeded (rate limit)")
                        print("   The API key is valid but has hit rate limits")
                        print("   Solutions: wait and retry, get new key, or upgrade to paid tier")
                    self.model = None
                elif "api" in error_str and ("key" in error_str or "invalid" in error_str or "expired" in error_str):
                    self.api_key_valid = False
                    self.api_key_error = str(e)
                    print(f"⚠️  Error initializing Gemini model: {str(e)[:100]}")
                    print("   Run: ./scripts/fix_api_key.sh to fix this")
                    self.api_key_set = False
                    self.model = None
                elif model_not_found:
                    self.api_key_valid = False
                    self.api_key_error = f"MODEL_NOT_FOUND: {str(e)}"
                    self.model = None
                    print("⚠️  Model name not supported/available for this API key/library.")
                    print(f"   Current model: {self.model_name}")
                    print("   Fix: set GEMINI_MODEL=gemini-2.0-flash and restart backend")
                else:
                    print(f"⚠️  Error initializing Gemini model: {e}")
                    self.api_key_set = False
                    self.model = None
        else:
            self.model = None
            
        self.conversation_history = []
        self.execution_log = []
        self.state = ExecutionState.READY

    def _fallback_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Heuristic fallback planner (no LLM):
        Extracts obvious tool calls from the query so the lesson can run even when Gemini is unavailable (quota=0).
        Returns a list of {"tool_name": str, "inputs": dict}.
        """
        q = query.strip()
        q_lower = q.lower()

        plan: List[Dict[str, Any]] = []

        # Weather: detect cities mentioned; fall back to "weather in X" pattern.
        known_cities = ["new york", "london", "tokyo", "sydney", "paris"]
        cities = []
        for city in known_cities:
            if city in q_lower:
                cities.append(city.title() if city != "new york" else "New York")

        if not cities:
            m = re.search(r"weather\s+(?:in|for)\s+([a-zA-Z][a-zA-Z\\s\\-]{1,40})", q, flags=re.IGNORECASE)
            if m:
                city = m.group(1).strip().strip("?.!,")
                if city:
                    cities.append(city)

        unit = "fahrenheit" if "fahrenheit" in q_lower or "°f" in q_lower else "celsius"
        for city in cities:
            plan.append({"tool_name": "get_weather", "inputs": {"city": city, "unit": unit}})

        # Revenue: Q3 2024 / quarter 3 2024 patterns.
        for quarter, year in re.findall(r"q([1-4])\\s*(20\\d{2})", q_lower):
            plan.append({"tool_name": "calculate_revenue", "inputs": {"year": int(year), "quarter": int(quarter)}})
        m = re.search(r"quarter\\s*([1-4])\\s*(20\\d{2})", q_lower)
        if m and not any(p["tool_name"] == "calculate_revenue" for p in plan):
            plan.append({"tool_name": "calculate_revenue", "inputs": {"year": int(m.group(2)), "quarter": int(m.group(1))}})

        # Product search: "search for X", "find X products", "products: X"
        m = re.search(r"(?:search\\s+for|find)\\s+([a-zA-Z0-9\\s\\-]{2,50})(?:\\s+products)?", q, flags=re.IGNORECASE)
        if m and ("product" in q_lower or "products" in q_lower or "search" in q_lower):
            term = m.group(1).strip().strip("?.!,")
            if term and term.lower() not in ("weather",):
                plan.append({"tool_name": "search_products", "inputs": {"query": term, "max_results": 5}})

        # Document search: if query mentions docs/knowledge base/RAG/document.
        if any(k in q_lower for k in ["document", "docs", "knowledge base", "rag", "architecture"]):
            plan.append({"tool_name": "document_search", "inputs": {"query": q, "top_k": 3}})

        # De-dupe identical calls (tool + inputs)
        seen = set()
        deduped = []
        for item in plan:
            key = (item["tool_name"], json.dumps(item["inputs"], sort_keys=True))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def _execute_fallback(self, query: str) -> Dict[str, Any]:
        """Execute fallback plan and return ChatResponse-compatible result."""
        start_time = time.time()
        self.state = ExecutionState.FALLBACK
        self.execution_log = []

        plan = self._fallback_plan(query)
        if not plan:
            self.state = ExecutionState.FAILED
            return {
                "success": False,
                "error": (
                    "🚫 LLM Unavailable (quota/billing) and fallback planner couldn't infer tool calls.\n\n"
                    "Try a more explicit query, e.g.:\n"
                    '- "What is the weather in Tokyo?"\n'
                    '- "Calculate revenue for Q3 2024"\n'
                    '- "Search products for laptop"\n'
                    "\nOr use POST /execute-tool directly."
                ),
                "turn_count": 0,
                "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                "execution_log": [],
                "state": self.state,
            }

        turn_count = 1
        for step in plan:
            tool_name = step["tool_name"]
            inputs = step["inputs"]
            log_entry = {
                "turn": turn_count,
                "function": tool_name,
                "inputs": inputs,
                "timestamp": datetime.now().isoformat(),
            }
            result = self.registry.execute(tool_name, inputs)
            log_entry["result"] = result
            self.execution_log.append(log_entry)

        self.state = ExecutionState.COMPLETE
        tools_used = ", ".join([f'{x["function"]}' for x in self.execution_log])
        return {
            "success": True,
            "answer": f"Fallback mode (no LLM): executed {len(self.execution_log)} tool call(s): {tools_used}.",
            "turn_count": 1,
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            "execution_log": self.execution_log,
            "state": self.state,
        }
    
    def _build_tool_declarations(self):
        """Build tool declarations for Gemini"""
        declarations = []
        for schema in self.registry.get_all_schemas():
            declarations.append(genai.protos.Tool(
                function_declarations=[genai.protos.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            name: genai.protos.Schema(
                                type=self._map_type(prop["type"]),
                                description=prop["description"]
                            )
                            for name, prop in schema["parameters"]["properties"].items()
                        },
                        required=schema["parameters"]["required"]
                    )
                )]
            ))
        return declarations
    
    def _map_type(self, type_str: str):
        """Map JSON schema types to Gemini types"""
        mapping = {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "number": genai.protos.Type.NUMBER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }
        return mapping.get(type_str, genai.protos.Type.STRING)
    
    async def execute_loop(self, query: str, max_turns: int = 10) -> Dict[str, Any]:
        """Main execution loop"""
        if not self.api_key_set or self.model is None:
            # If LLM is unavailable due to quota/billing, optionally fall back to a heuristic tool planner
            if self.fallback_enabled:
                return self._execute_fallback(query)

            if not self.api_key_set:
                error_msg = (
                    "🔑 API Key not configured!\n\n"
                    "📝 Quick Fix:\n"
                    "   Run: ./scripts/fix_api_key.sh\n\n"
                    "This interactive script will guide you through:\n"
                    "   • Getting a new API key from Google\n"
                    "   • Setting it up automatically\n"
                    "   • Validating and restarting the backend\n\n"
                    "Or fix manually:\n"
                    "   1. Get API key: https://makersuite.google.com/app/apikey\n"
                    "   2. cd backend && echo 'GEMINI_API_KEY=your_key' > .env\n"
                    "   3. Restart: pkill -f 'python main.py' && cd backend && source venv/bin/activate && python main.py\n"
                )
            elif not self.api_key_valid:
                # Check if it's a quota error
                if self.api_key_error and ("QUOTA_EXCEEDED" in self.api_key_error or "QUOTA_ZERO" in self.api_key_error):
                    if "QUOTA_ZERO" in self.api_key_error:
                        error_msg = (
                            "🚫 No Free-Tier Quota For This Project (limit=0)\n\n"
                            "Your API key can list models, but this project has **0 free-tier quota** for `generateContent`.\n"
                            "This is a plan/billing/project configuration issue—not something that will resolve by waiting.\n\n"
                            "📝 Fix options:\n"
                            "   1. Create a NEW API key in a NEW project (AI Studio): https://makersuite.google.com/app/apikey\n"
                            "      (choose “Create API key in new project”)\n"
                            "   2. Enable billing / upgrade plan for this project\n"
                            "   3. Check usage/quota: https://ai.dev/usage?tab=rate-limit\n\n"
                            "Meanwhile, use /execute-tool for tool testing (no LLM needed).\n\n"
                        )
                        error_msg += f"Error: {self.api_key_error[11:260]}...\n"
                    else:
                        error_msg = (
                            "⏱️  API Quota Exceeded (Rate Limit)\n\n"
                            "Your API key is valid but has hit rate limits.\n\n"
                            "📝 Solutions:\n"
                            "   1. ⏰ Wait and retry (rate limits reset)\n"
                            "   2. 🔑 Get a NEW API key: https://makersuite.google.com/app/apikey\n"
                            "      (Create a new key in a different project)\n"
                            "   3. 💳 Upgrade to paid tier for higher limits\n"
                            "   4. 🧪 Use /execute-tool endpoint for direct tool testing (no LLM needed)\n\n"
                        )
                        error_msg += f"Error: {self.api_key_error[15:260]}...\n"
                elif self.api_key_error and "MODEL_NOT_FOUND" in self.api_key_error:
                    error_msg = (
                        "🚫 Model Not Found / Not Supported\n\n"
                        f"The configured model `{self.model_name}` is not supported for `generateContent` with this API setup.\n\n"
                        "📝 Fix:\n"
                        "   export GEMINI_MODEL=gemini-2.0-flash\n"
                        "   (then restart the backend)\n\n"
                        "Meanwhile, use /execute-tool for tool testing (no LLM needed).\n\n"
                    )
                    error_msg += f"Error: {self.api_key_error[15:260]}...\n"
                else:
                    error_msg = (
                        "🔑 API Key is invalid or expired!\n\n"
                        "📝 Quick Fix:\n"
                        "   Run: ./scripts/fix_api_key.sh\n\n"
                        "This interactive script will guide you through:\n"
                        "   • Getting a new API key from Google\n"
                        "   • Setting it up automatically\n"
                        "   • Validating and restarting the backend\n"
                    )
                    if self.api_key_error:
                        error_msg += f"\nError details: {self.api_key_error[:150]}\n"
            else:
                error_msg = (
                    "🔑 API Key configuration issue!\n\n"
                    "📝 Quick Fix:\n"
                    "   Run: ./scripts/fix_api_key.sh\n\n"
                    "Check status: curl http://localhost:8000/api-key-status\n"
                )
            
            return {
                "success": False,
                "error": error_msg,
                "turn_count": 0,
                "execution_time_ms": 0.0,
                "execution_log": [],
                "state": ExecutionState.FAILED
            }
        
        self.state = ExecutionState.READY
        self.conversation_history = []
        self.execution_log = []
        
        start_time = time.time()
        turn_count = 0
        
        # Initial query
        self.conversation_history.append({"role": "user", "parts": [query]})
        
        while turn_count < max_turns:
            turn_count += 1
            self.state = ExecutionState.CALLING_LLM
            
            try:
                # Call LLM
                response = self.model.generate_content(
                    self.conversation_history,
                    tools=self._build_tool_declarations()
                )
                
                self.state = ExecutionState.PARSING_RESPONSE
                
                # Check for tool calls
                if not response.candidates[0].content.parts:
                    self.state = ExecutionState.FAILED
                    return {
                        "success": False,
                        "error": "Empty response from LLM",
                        "turn_count": turn_count,
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                        "execution_log": self.execution_log,
                        "state": self.state
                    }
                
                # Extract parts
                parts = response.candidates[0].content.parts
                has_function_calls = any(hasattr(part, 'function_call') for part in parts)
                
                if not has_function_calls:
                    # Final answer
                    text_parts = [part.text for part in parts if hasattr(part, 'text')]
                    final_answer = " ".join(text_parts)
                    
                    self.state = ExecutionState.COMPLETE
                    return {
                        "success": True,
                        "answer": final_answer,
                        "turn_count": turn_count,
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                        "execution_log": self.execution_log,
                        "state": self.state
                    }
                
                # Execute function calls
                self.state = ExecutionState.EXECUTING_TOOLS
                function_responses = []
                
                for part in parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        func_name = fc.name
                        func_args = dict(fc.args)
                        
                        self.execution_log.append({
                            "turn": turn_count,
                            "function": func_name,
                            "inputs": func_args,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Execute function
                        result = self.registry.execute(func_name, func_args)
                        
                        # Build response
                        function_responses.append(
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=func_name,
                                    response={"result": result}
                                )
                            )
                        )
                        
                        self.execution_log[-1]["result"] = result
                
                # Add function call and responses to history
                self.state = ExecutionState.BUILDING_CONTEXT
                self.conversation_history.append({"role": "model", "parts": parts})
                self.conversation_history.append({"role": "user", "parts": function_responses})
                
            except Exception as e:
                self.state = ExecutionState.FAILED
                error_msg = str(e)
                error_str_lower = error_msg.lower()
                
                # Check for quota/rate limit errors first (429)
                if "429" in error_msg or "quota" in error_str_lower or "rate limit" in error_str_lower:
                    if "limit: 0" in error_msg:
                        error_msg = (
                            "🚫 Model Not Available On Current Tier\n\n"
                            f"The configured model `{self.model_name}` appears to have free-tier limit=0.\n\n"
                            "📝 Fix:\n"
                            "   1. Set a generally-available model:\n"
                            "      export GEMINI_MODEL=gemini-1.5-flash\n"
                            "   2. Restart the backend\n\n"
                            "Meanwhile, use /execute-tool endpoint (no LLM needed).\n\n"
                            f"Error: {error_msg[:300]}"
                        )
                    else:
                        error_msg = (
                            "⏱️  API Quota Exceeded (Rate Limit)\n\n"
                            "Your API key has hit rate limits.\n\n"
                            "📝 Solutions:\n"
                            "   1. ⏰ Wait and retry (rate limits reset)\n"
                            "   2. 🔑 Get a NEW API key: https://makersuite.google.com/app/apikey\n"
                            "      (Create new key in different project)\n"
                            "   3. 💳 Upgrade to paid tier for higher limits\n"
                            "   4. 🧪 Meanwhile, use /execute-tool endpoint (no LLM needed)\n\n"
                            "To set new API key: ./scripts/fix_api_key.sh\n\n"
                            f"Error: {error_msg[:300]}"
                        )
                # Check for API key errors (various formats Google might use)
                elif any(keyword in error_str_lower for keyword in [
                    "api key", "api_key", "apikey", "api_key_invalid", 
                    "api key expired", "api key invalid", "invalid api key"
                ]):
                    error_msg = (
                        "🔑 API Key Error: Your Google Gemini API key is invalid or expired.\n\n"
                        "📝 Quick Fix:\n"
                        "   Run: ./scripts/fix_api_key.sh\n\n"
                        "This interactive script will:\n"
                        "   • Guide you to get a new API key\n"
                        "   • Validate it automatically\n"
                        "   • Restart the backend for you\n\n"
                        "Or fix manually:\n"
                        "   1. Get API key: https://makersuite.google.com/app/apikey\n"
                        "   2. cd backend && echo 'GEMINI_API_KEY=your_key' > .env\n"
                        "   3. Restart: pkill -f 'python main.py' && cd backend && source venv/bin/activate && python main.py\n\n"
                        "Check status: curl http://localhost:8000/api-key-status\n\n"
                        f"Original error: {error_msg[:200]}"
                    )
                
                return {
                    "success": False,
                    "error": error_msg,
                    "turn_count": turn_count,
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                    "execution_log": self.execution_log,
                    "state": self.state
                }
        
        # Max turns exceeded
        self.state = ExecutionState.FAILED
        return {
            "success": False,
            "error": f"Maximum turns ({max_turns}) exceeded",
            "turn_count": turn_count,
            "execution_time_ms": round((time.time() - start_time) * 1000, 2),
            "execution_log": self.execution_log,
            "state": self.state
        }

# Initialize executor
executor = ToolExecutor(registry)

# ============================================================================
# API ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    query: str
    max_turns: int = 10

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    turn_count: int
    execution_time_ms: float
    execution_log: List[Dict[str, Any]]
    state: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Execute tool loop for user query"""
    try:
        result = await executor.execute_loop(request.query, request.max_turns)
        # Ensure result has all required fields for ChatResponse (safety check)
        if "execution_log" not in result:
            result["execution_log"] = []
        if "execution_time_ms" not in result:
            result["execution_time_ms"] = 0.0
        if "state" not in result:
            result["state"] = ExecutionState.FAILED
        return ChatResponse(**result)
    except Exception as e:
        # Return proper JSON error response instead of letting FastAPI return plain text
        error_msg = str(e)
        error_str_lower = error_msg.lower()
        
        # Check for quota/rate limit errors first (429)
        if "429" in error_msg or "quota" in error_str_lower or "rate limit" in error_str_lower:
            error_msg = (
                "⏱️  API Quota Exceeded (Rate Limit)\n\n"
                "Your API key has hit the free tier rate limits.\n\n"
                "📝 Solutions:\n"
                "   1. ⏰ Wait ~1 hour for quota to reset\n"
                "   2. 🔑 Get a NEW API key: https://makersuite.google.com/app/apikey\n"
                "      (Create new key in different project)\n"
                "   3. 💳 Upgrade to paid tier for higher limits\n"
                "   4. 🧪 Meanwhile, use /execute-tool endpoint (no LLM needed)\n\n"
                "To set new API key: ./scripts/fix_api_key.sh\n\n"
                f"Error: {error_msg[:300]}"
            )
        # Check for API key errors (various formats Google might use)
        elif any(keyword in error_str_lower for keyword in [
            "api key", "api_key", "apikey", "api_key_invalid", 
            "api key expired", "api key invalid", "invalid api key"
        ]):
            error_msg = (
                "🔑 API Key Error: Your Google Gemini API key is invalid or expired.\n\n"
                "📝 Quick Fix:\n"
                "   Run: ./scripts/fix_api_key.sh\n\n"
                "This interactive script will:\n"
                "   • Guide you to get a new API key\n"
                "   • Validate it automatically\n"
                "   • Restart the backend for you\n\n"
                "Or fix manually:\n"
                "   1. Get API key: https://makersuite.google.com/app/apikey\n"
                "   2. cd backend && echo 'GEMINI_API_KEY=your_key' > .env\n"
                "   3. Restart: pkill -f 'python main.py' && cd backend && source venv/bin/activate && python main.py\n\n"
                "Check status: curl http://localhost:8000/api-key-status\n\n"
                f"Original error: {error_msg[:200]}"
            )
        
        error_response = {
            "success": False,
            "error": error_msg,
            "turn_count": 0,
            "execution_time_ms": 0.0,
            "execution_log": [],
            "state": ExecutionState.FAILED
        }
        # Log the full traceback for debugging
        print(f"Error in chat endpoint: {traceback.format_exc()}")
        return ChatResponse(**error_response)

@app.get("/tools")
async def get_tools():
    """Get registered tool schemas"""
    return {
        "tools": registry.get_all_schemas(),
        "count": len(registry._functions)
    }

@app.get("/stats")
async def get_stats():
    """Get execution statistics"""
    return registry.get_stats()

class ExecuteToolRequest(BaseModel):
    tool_name: str
    inputs: Dict[str, Any]

@app.post("/execute-tool")
async def execute_tool(request: ExecuteToolRequest):
    """Execute a tool directly without LLM (for testing)"""
    if request.tool_name not in registry._functions:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{request.tool_name}' not found. Available tools: {list(registry._functions.keys())}"
        )
    
    result = registry.execute(request.tool_name, request.inputs)
    return result

@app.get("/health")
async def health():
    """Health check endpoint"""
    api_key_status = "configured" if GEMINI_API_KEY != "NOT_SET" else "not_configured"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "registered_tools": len(registry._functions),
        "api_key_status": api_key_status,
        "api_key_valid": executor.api_key_valid if hasattr(executor, 'api_key_valid') else False,
        "llm_enabled": executor.model is not None,
        "fallback_enabled": executor.fallback_enabled if hasattr(executor, "fallback_enabled") else False,
        "model": GEMINI_MODEL,
    }

@app.get("/api-key-status")
async def api_key_status():
    """Check API key status and provide setup instructions"""
    has_key = GEMINI_API_KEY != "NOT_SET"
    is_valid = False
    error_msg = None
    is_quota_error = False
    is_model_unavailable = False
    is_quota_zero = False
    
    if has_key:
        # Try to validate the key
        try:
            test_model = genai.GenerativeModel(GEMINI_MODEL)
            test_response = test_model.generate_content("test")
            is_valid = True
        except Exception as e:
            error_msg = str(e)
            error_str_lower = str(e).lower()
            # Check if it's a quota/rate limit error
            if "429" in str(e) or "quota" in error_str_lower or "rate limit" in error_str_lower:
                is_quota_error = True
                if "limit: 0" in str(e):
                    if "free_tier" in str(e):
                        is_quota_zero = True
            # Check if model name is unsupported/unavailable
            if (
                "is not found for api version" in error_str_lower
                or ("models/" in error_str_lower and "not found" in error_str_lower)
                or ("not supported for generatecontent" in error_str_lower)
            ):
                is_model_unavailable = True
            is_valid = False
    
    if is_quota_zero:
        setup_instructions = {
            "issue": "Project has 0 free-tier quota for generateContent (limit=0)",
            "fix1": "Create a NEW API key in a NEW project (AI Studio): https://makersuite.google.com/app/apikey",
            "fix2": "Enable billing / upgrade plan for this project",
            "usage": "Check usage/quota: https://ai.dev/usage?tab=rate-limit",
            "note": "Meanwhile, use /execute-tool endpoint for testing (no API key needed)",
        }
    elif is_model_unavailable:
        setup_instructions = {
            "issue": "Model not available/supported for this API setup",
            "current_model": GEMINI_MODEL,
            "fix1": "Set a generally-available model: export GEMINI_MODEL=gemini-2.0-flash",
            "fix2": "Restart backend after updating env (.env or exported var)",
            "note": "Meanwhile, use /execute-tool endpoint for testing (no API key needed)",
        }
    elif is_quota_error:
        setup_instructions = {
            "issue": "API quota exceeded (free tier rate limit)",
            "solution1": "⏰ Wait and retry (rate limits reset)",
            "solution2": "🔑 Get NEW API key: https://makersuite.google.com/app/apikey (different project)",
            "solution3": "💳 Upgrade to paid tier for higher limits",
            "solution4": "🧪 Use /execute-tool endpoint for testing (no API key needed)",
            "to_set_new_key": "Run: ./scripts/fix_api_key.sh"
        }
    else:
        setup_instructions = {
            "step1": "Get API key from: https://makersuite.google.com/app/apikey",
            "step2": "Run: ./scripts/fix_api_key.sh (interactive, recommended)",
            "step3": "Or manually: cd backend && echo 'GEMINI_API_KEY=your_key' > .env",
            "step4": "Restart backend: pkill -f 'python main.py' && cd backend && source venv/bin/activate && python main.py"
        }
    
    return {
        "has_api_key": has_key,
        "is_valid": is_valid,
        "is_quota_error": is_quota_error,
        "is_model_unavailable": is_model_unavailable,
        "is_quota_zero": is_quota_zero,
        "model": GEMINI_MODEL,
        "error": error_msg,
        "setup_instructions": setup_instructions,
        "quick_fix": (
            "Create a NEW API key in a NEW project, or enable billing"
            if is_quota_zero
            else (
                "Set GEMINI_MODEL=gemini-2.0-flash and restart backend"
                if is_model_unavailable
                else ("Wait and retry, or get new API key" if is_quota_error else "Run: ./scripts/fix_api_key.sh")
            )
        )
    }

# WebSocket for real-time monitoring
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial stats
        await websocket.send_json(registry.get_stats())
        
        while True:
            # Send updates every 2 seconds
            await asyncio.sleep(2)
            try:
                stats = registry.get_stats()
                await websocket.send_json(stats)
            except:
                break
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting L18 Tool Execution Loop Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
