"""
FastAPI Application with Gemini AI Tool Calling
Production-grade schema validation and error handling
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from tool_registry import registry
from schemas import TOOL_SCHEMAS
import tools
import os
from typing import Optional
from collections import defaultdict
from datetime import datetime
import json

app = FastAPI(title="L17 Tool Schema Validator")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics tracking
metrics = {
    "total_queries": 0,
    "total_tool_calls": 0,
    "total_validations": 0,
    "tool_usage": defaultdict(int),
    "validation_success": 0,
    "validation_failures": 0,
    "query_success": 0,
    "query_failures": 0,
    "start_time": datetime.now().isoformat()
}

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Query endpoint will not work.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize tool registry
for tool_name, schema_info in TOOL_SCHEMAS.items():
    function = getattr(tools, tool_name)
    registry.register_tool(
        name=tool_name,
        input_schema=schema_info["input"],
        output_schema=schema_info["output"],
        function=function,
        description=schema_info["description"]
    )

class QueryRequest(BaseModel):
    query: str

class ValidationRequest(BaseModel):
    tool_name: str
    parameters: dict

@app.get("/")
def root():
    return {
        "service": "L17 Tool Schema Validator",
        "status": "running",
        "tools_registered": len(registry.tools)
    }

@app.get("/tools")
def list_tools():
    """List all registered tools with schemas"""
    tools_info = []
    for tool_name in registry.tools:
        tools_info.append(registry.get_tool_info(tool_name))
    return {
        "tools": tools_info,
        "count": len(tools_info)
    }

@app.get("/metrics")
def get_metrics():
    """Get dashboard metrics"""
    return {
        "total_queries": metrics["total_queries"],
        "total_tool_calls": metrics["total_tool_calls"],
        "total_validations": metrics["total_validations"],
        "tool_usage": dict(metrics["tool_usage"]),
        "validation_success": metrics["validation_success"],
        "validation_failures": metrics["validation_failures"],
        "query_success": metrics["query_success"],
        "query_failures": metrics["query_failures"],
        "success_rate": (
            metrics["validation_success"] / max(metrics["total_validations"], 1) * 100
        ) if metrics["total_validations"] > 0 else 0,
        "uptime_seconds": (datetime.now() - datetime.fromisoformat(metrics["start_time"])).total_seconds(),
        "start_time": metrics["start_time"]
    }

@app.post("/validate")
def validate_tool_call(request: ValidationRequest):
    """
    Validate tool parameters without execution
    Useful for debugging and schema testing
    """
    metrics["total_validations"] += 1
    
    if request.tool_name not in registry.tools:
        metrics["validation_failures"] += 1
        raise HTTPException(404, f"Tool not found: {request.tool_name}")
    
    tool = registry.tools[request.tool_name]
    input_schema = tool["input_schema"]
    
    try:
        validated = input_schema(**request.parameters)
        metrics["validation_success"] += 1
        return {
            "valid": True,
            "tool": request.tool_name,
            "validated_parameters": validated.model_dump(),
            "schema": input_schema.model_json_schema()
        }
    except ValidationError as e:
        metrics["validation_failures"] += 1
        return {
            "valid": False,
            "tool": request.tool_name,
            "errors": e.errors(),
            "schema": input_schema.model_json_schema()
        }

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Process natural language query with Gemini AI
    Demonstrates end-to-end tool calling with validation
    """
    metrics["total_queries"] += 1
    
    # Check if API key is configured
    if not GEMINI_API_KEY:
        metrics["query_failures"] += 1
        raise HTTPException(
            status_code=400,
            detail={
                "error": "API key not configured",
                "message": "GEMINI_API_KEY is not set in the .env file. Please add a valid Gemini API key.",
                "instructions": "Get your API key from: https://makersuite.google.com/app/apikey"
            }
        )
    
    try:
        # Get Gemini tools
        gemini_tools = registry.get_gemini_tools()
        
        # Create model with tools
        # Use actual available models from the API
        # Based on available models: gemini-2.5-flash, gemini-2.0-flash-exp, etc.
        models_to_try = [
            'gemini-2.5-flash',      # Latest stable flash model
            'gemini-2.0-flash-exp',  # Experimental version
            'gemini-2.0-flash',      # Stable 2.0 version
            'gemini-2.0-flash-001'   # Specific version
        ]
        
        model = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    tools=gemini_tools
                )
                # If we get here, model was created successfully
                break
            except Exception as e:
                last_error = e
                # Continue to next model
                continue
        
        if model is None:
            raise Exception(f"Failed to initialize any Gemini model. Tried: {', '.join(models_to_try)}. Last error: {last_error}")
        
        # Start chat
        chat = model.start_chat()
        
        # Send query
        response = chat.send_message(request.query)
        
        # Process tool calls and responses
        tool_results = []
        final_response = None
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            function_calls = []
            text_parts = []
            
            # Process all parts of the response
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            
            # If we have text, collect it
            if text_parts:
                final_response = " ".join(text_parts)
            
            # If we have function calls, execute them and send results back
            if function_calls:
                function_responses = []
                
                for fc in function_calls:
                    if not fc.name:
                        continue
                        
                    metrics["total_tool_calls"] += 1
                    metrics["tool_usage"][fc.name] += 1
                    
                    # Convert function call args to dict
                    try:
                        fc_args_dict = MessageToDict(fc.args, preserving_proto_field_name=True)
                        if isinstance(fc_args_dict, dict):
                            fc_args_dict = {k: v for k, v in fc_args_dict.items() if v is not None}
                    except Exception as e:
                        try:
                            fc_args_dict = dict(fc.args) if hasattr(fc.args, 'keys') else {}
                        except:
                            fc_args_dict = {}
                            print(f"Warning: Could not convert function args: {e}")
                    
                    # Validate and execute
                    result = registry.validate_and_execute(
                        fc.name,
                        fc_args_dict
                    )
                    
                    tool_results.append({
                        "tool": fc.name,
                        "parameters": fc_args_dict,
                        "result": result
                    })
                    
                    # Prepare function response
                    if isinstance(result, dict) and "result" in result and result.get("success"):
                        response_data = result["result"]
                    else:
                        response_data = result
                    
                    # Convert dict to protobuf Struct
                    struct_value = struct_pb2.Struct()
                    struct_value.update(response_data)
                    
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc.name,
                                response=struct_value
                            )
                        )
                    )
                
                # Send all function responses back to Gemini
                if function_responses:
                    response = chat.send_message(
                        genai.protos.Content(parts=function_responses)
                    )
                else:
                    break
            else:
                # No more function calls, we're done
                break
        
        metrics["query_success"] += 1
        return {
            "query": request.query,
            "tool_calls": tool_results,
            "response": final_response or "No response generated",
            "tools_available": len(gemini_tools)
        }
        
    except Exception as e:
        metrics["query_failures"] += 1
        error_str = str(e)
        
        # Handle quota/rate limit errors with helpful suggestions
        if "quota" in error_str.lower() or "429" in error_str or "rate limit" in error_str.lower():
            # Extract retry delay if available
            retry_seconds = None
            if "retry_delay" in error_str or "retry in" in error_str.lower():
                import re
                retry_match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                if retry_match:
                    retry_seconds = float(retry_match.group(1))
            
            detail_response = {
                "error": "Quota exceeded",
                "message": "You have exceeded your Gemini API quota. The system will automatically retry with alternative models.",
                "suggestions": [
                    "1. Wait for quota to reset (usually daily)",
                    "2. Check your usage: https://ai.dev/usage?tab=rate-limit",
                    "3. Review rate limits: https://ai.google.dev/gemini-api/docs/rate-limits",
                    "4. Consider upgrading your plan for higher limits",
                    "5. Use the /validate endpoint which doesn't require API calls"
                ],
                "working_endpoints": [
                    "GET /health - Always available",
                    "GET /tools - Always available", 
                    "POST /validate - No API quota needed",
                    "GET /metrics - Always available"
                ]
            }
            
            if retry_seconds:
                detail_response["retry_after_seconds"] = retry_seconds
                detail_response["retry_after_readable"] = f"{int(retry_seconds)} seconds"
            
            raise HTTPException(
                status_code=429,
                detail=detail_response
            )
        
        # Handle specific API key errors
        if "API key" in error_str or "API_KEY" in error_str or "InvalidArgument" in error_str:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "API key issue",
                    "message": "The Gemini API key is expired or invalid. Please update the GEMINI_API_KEY in the .env file.",
                    "instructions": [
                        "1. Get a new API key from: https://makersuite.google.com/app/apikey",
                        "2. Run: ./update_api_key.sh (or manually edit backend/.env)",
                        "3. Restart the backend service: ./stop.sh && ./start.sh"
                    ],
                    "quick_fix": "Run './update_api_key.sh' from the project root directory",
                    "original_error": error_str
                }
            )
        
        # Handle other errors
        import traceback
        error_details = traceback.format_exc()
        print(f"Query processing error: {error_str}")
        print(f"Traceback: {error_details}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": error_str
            }
        )

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "tools_count": len(registry.tools)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
