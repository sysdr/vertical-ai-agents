"""
L8 Core Agent Theory: Decision Maker Implementation
Enterprise-grade autonomous agent with planning, deliberation, and tool orchestration
"""
import asyncio
import json
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# ============================================================================
# Models
# ============================================================================

class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"

class ToolType(str, Enum):
    CALCULATOR = "calculator"
    WEATHER = "weather"
    DATABASE = "database"
    SEARCH = "search"

class ToolDefinition(BaseModel):
    name: str
    type: ToolType
    description: str
    parameters: Dict[str, Any]

class Step(BaseModel):
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    description: str

class Plan(BaseModel):
    goal: str
    steps: List[Step]
    estimated_steps: int
    confidence: float = Field(ge=0.0, le=1.0)

class DecisionTrace(BaseModel):
    timestamp: datetime
    state: AgentState
    reasoning: str
    action: str
    input_context: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    duration_ms: float

class AgentGoal(BaseModel):
    goal: str
    context: Dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 10
    timeout_seconds: int = 30

class AgentResponse(BaseModel):
    goal: str
    state: AgentState
    plan: Optional[Plan] = None
    results: Dict[str, Any]
    traces: List[DecisionTrace]
    total_duration_ms: float
    success: bool

# ============================================================================
# Tool Registry
# ============================================================================

class Tool:
    """Base tool interface"""
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class CalculatorTool(Tool):
    """Performs mathematical calculations"""
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        operation = parameters.get("operation")
        operands = parameters.get("operands", [])
        
        try:
            if operation == "add":
                result = sum(operands)
            elif operation == "multiply":
                result = 1
                for n in operands:
                    result *= n
            elif operation == "factorial":
                n = operands[0]
                result = 1
                for i in range(1, n + 1):
                    result *= i
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            
            return {
                "success": True,
                "result": result,
                "operation": operation,
                "operands": operands
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class WeatherTool(Tool):
    """Simulates weather API calls"""
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        location = parameters.get("location", "unknown")
        # Simulate API call
        await asyncio.sleep(0.1)
        return {
            "success": True,
            "location": location,
            "temperature": 72,
            "conditions": "sunny",
            "humidity": 45
        }

class DatabaseTool(Tool):
    """Simulates database queries"""
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        query = parameters.get("query", "")
        # Simulate DB query
        await asyncio.sleep(0.15)
        return {
            "success": True,
            "query": query,
            "rows_affected": 5,
            "data": [{"id": 1, "value": "sample"}]
        }

class SearchTool(Tool):
    """Simulates web search"""
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        query = parameters.get("query", "")
        await asyncio.sleep(0.2)
        return {
            "success": True,
            "query": query,
            "results": [
                {"title": "Result 1", "snippet": "Sample content"},
                {"title": "Result 2", "snippet": "More content"}
            ]
        }

class ToolRegistry:
    """Manages available tools for agent"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Register default tools"""
        tools_config = [
            ToolDefinition(
                name="calculator",
                type=ToolType.CALCULATOR,
                description="Performs mathematical operations",
                parameters={"operation": "str", "operands": "list"}
            ),
            ToolDefinition(
                name="weather",
                type=ToolType.WEATHER,
                description="Gets weather information",
                parameters={"location": "str"}
            ),
            ToolDefinition(
                name="database",
                type=ToolType.DATABASE,
                description="Queries database",
                parameters={"query": "str"}
            ),
            ToolDefinition(
                name="search",
                type=ToolType.SEARCH,
                description="Searches the web",
                parameters={"query": "str"}
            )
        ]
        
        for tool_def in tools_config:
            if tool_def.type == ToolType.CALCULATOR:
                self.tools[tool_def.name] = CalculatorTool(tool_def)
            elif tool_def.type == ToolType.WEATHER:
                self.tools[tool_def.name] = WeatherTool(tool_def)
            elif tool_def.type == ToolType.DATABASE:
                self.tools[tool_def.name] = DatabaseTool(tool_def)
            elif tool_def.type == ToolType.SEARCH:
                self.tools[tool_def.name] = SearchTool(tool_def)
    
    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        return [tool.definition for tool in self.tools.values()]

# ============================================================================
# Decision Maker Core
# ============================================================================

class DecisionMaker:
    """Core agent decision-making engine"""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        # Use gemini-1.5-flash which is available on free tier
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_decision_time = int(os.getenv("MAX_DECISION_TIME", "5"))
    
    async def gemini_json_call(self, prompt: str, timeout: int = 5) -> Dict[str, Any]:
        """Reuses L7's JSON-structured LLM call"""
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.model.generate_content, prompt),
                timeout=timeout
            )
            
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            return json.loads(text.strip())
        except asyncio.TimeoutError:
            logger.warning("llm_timeout", timeout=timeout)
            return {"error": "LLM timeout", "success": False}
        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e), text=text[:200])
            return {"error": "Invalid JSON", "success": False}
        except Exception as e:
            error_msg = str(e)
            # Check for quota errors
            if "429" in error_msg or "quota" in error_msg.lower() or "Quota exceeded" in error_msg:
                logger.error("api_quota_exceeded", error=error_msg[:200])
                return {
                    "error": "API quota exceeded. Please check your Gemini API quota or wait before retrying.",
                    "success": False,
                    "quota_error": True
                }
            logger.error("llm_api_error", error=error_msg[:200])
            return {"error": f"API error: {error_msg[:100]}", "success": False}
    
    def _generate_fallback_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Generate a fallback plan using pattern matching when LLM is unavailable"""
        goal_lower = goal.lower()
        steps = []
        
        # Pattern matching for common goals
        if "factorial" in goal_lower or "!" in goal:
            # Extract number from goal
            import re
            numbers = re.findall(r'\d+', goal)
            if numbers:
                num = int(numbers[0])
                steps.append(Step(
                    step_id="step_1",
                    tool_name="calculator",
                    parameters={"operation": "factorial", "operands": [num]},
                    description=f"Calculate {num} factorial"
                ))
        
        elif "weather" in goal_lower:
            # Extract location
            location = "San Francisco"  # default
            if "for" in goal_lower:
                parts = goal_lower.split("for")
                if len(parts) > 1:
                    location = parts[1].strip()
            steps.append(Step(
                step_id="step_1",
                tool_name="weather",
                parameters={"location": location},
                description=f"Get weather for {location}"
            ))
        
        elif "search" in goal_lower:
            query = goal.replace("search for", "").replace("Search for", "").strip()
            if not query:
                query = goal
            steps.append(Step(
                step_id="step_1",
                tool_name="search",
                parameters={"query": query},
                description=f"Search for: {query}"
            ))
        
        elif "database" in goal_lower or "query" in goal_lower:
            query = goal
            steps.append(Step(
                step_id="step_1",
                tool_name="database",
                parameters={"query": query},
                description=f"Query database: {query}"
            ))
        
        elif "calculate" in goal_lower or "add" in goal_lower or "multiply" in goal_lower:
            # Extract numbers and operation
            import re
            numbers = [int(n) for n in re.findall(r'\d+', goal)]
            if numbers:
                if "add" in goal_lower or "+" in goal:
                    steps.append(Step(
                        step_id="step_1",
                        tool_name="calculator",
                        parameters={"operation": "add", "operands": numbers},
                        description=f"Add numbers: {numbers}"
                    ))
                elif "multiply" in goal_lower or "*" in goal or "×" in goal:
                    steps.append(Step(
                        step_id="step_1",
                        tool_name="calculator",
                        parameters={"operation": "multiply", "operands": numbers},
                        description=f"Multiply numbers: {numbers}"
                    ))
                else:
                    # Default to add
                    steps.append(Step(
                        step_id="step_1",
                        tool_name="calculator",
                        parameters={"operation": "add", "operands": numbers},
                        description=f"Calculate with numbers: {numbers}"
                    ))
        
        # If no pattern matched, create a default calculator step
        if not steps:
            steps.append(Step(
                step_id="step_1",
                tool_name="calculator",
                parameters={"operation": "add", "operands": [1, 2]},
                description="Default calculation step"
            ))
        
        return Plan(
            goal=goal,
            steps=steps,
            estimated_steps=len(steps),
            confidence=0.7
        )
    
    async def generate_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Generates execution plan for goal"""
        start = time.time()
        
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}" 
            for t in self.tool_registry.list_tools()
        ])
        
        prompt = f"""You are an AI agent planner. Given a goal and available tools, create a step-by-step execution plan.

Goal: {goal}

Available Tools:
{tools_desc}

Context: {json.dumps(context)}

Return a JSON object with this structure:
{{
    "steps": [
        {{
            "step_id": "step_1",
            "tool_name": "calculator",
            "parameters": {{"operation": "add", "operands": [1, 2]}},
            "description": "Add numbers"
        }}
    ],
    "confidence": 0.95
}}

Make the plan concrete and executable. Each step must use a valid tool."""
        
        result = await self.gemini_json_call(prompt, timeout=self.max_decision_time)
        
        if "error" in result or not result.get("steps"):
            # Use fallback plan generator
            logger.info("using_fallback_plan", goal=goal, reason="LLM unavailable or returned empty plan")
            fallback_plan = self._generate_fallback_plan(goal, context)
            duration = (time.time() - start) * 1000
            logger.info("fallback_plan_generated", goal=goal, steps=len(fallback_plan.steps), duration_ms=duration)
            return fallback_plan
        
        steps = [Step(**step) for step in result.get("steps", [])]
        duration = (time.time() - start) * 1000
        
        logger.info("plan_generated", goal=goal, steps=len(steps), duration_ms=duration)
        
        return Plan(
            goal=goal,
            steps=steps,
            estimated_steps=len(steps),
            confidence=result.get("confidence", 0.8)
        )
    
    async def deliberate_step(self, step: Step, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deliberates on executing a single step"""
        start = time.time()
        
        prompt = f"""You are deliberating on executing this step:

Step: {step.description}
Tool: {step.tool_name}
Parameters: {json.dumps(step.parameters)}
Current Context: {json.dumps(context)}

Should this step be executed? Are the parameters correct?

Return JSON:
{{
    "should_execute": true/false,
    "reasoning": "explanation",
    "modified_parameters": {step.parameters} // if changes needed
}}"""
        
        result = await self.gemini_json_call(prompt, timeout=self.max_decision_time)
        duration = (time.time() - start) * 1000
        
        # If API fails, default to executing the step
        if "error" in result:
            logger.info("deliberation_fallback", step_id=step.step_id, reason="LLM unavailable, defaulting to execute")
            result = {
                "should_execute": True,
                "reasoning": "LLM unavailable, proceeding with execution",
                "modified_parameters": step.parameters
            }
        
        logger.info("step_deliberation", step_id=step.step_id, duration_ms=duration)
        
        return result
    
    async def execute_step(
        self, 
        step: Step, 
        context: Dict[str, Any]
    ) -> tuple[Dict[str, Any], DecisionTrace]:
        """Executes a single step with tracing"""
        start = time.time()
        trace_start = datetime.now()
        
        # Get tool
        tool = self.tool_registry.get(step.tool_name)
        if not tool:
            error_result = {"success": False, "error": f"Tool not found: {step.tool_name}"}
            trace = DecisionTrace(
                timestamp=trace_start,
                state=AgentState.FAILED,
                reasoning=f"Tool {step.tool_name} not available",
                action=f"execute_{step.tool_name}",
                input_context={"step": step.dict()},
                output=error_result,
                duration_ms=(time.time() - start) * 1000
            )
            return error_result, trace
        
        # Execute with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    tool.execute(step.parameters),
                    timeout=10
                )
                
                trace = DecisionTrace(
                    timestamp=trace_start,
                    state=AgentState.EXECUTING,
                    reasoning=f"Executed {step.tool_name} with {step.parameters}",
                    action=f"execute_{step.tool_name}",
                    input_context={"step": step.dict(), "attempt": attempt + 1},
                    output=result,
                    duration_ms=(time.time() - start) * 1000
                )
                
                logger.info(
                    "step_executed",
                    step_id=step.step_id,
                    tool=step.tool_name,
                    success=result.get("success"),
                    attempt=attempt + 1
                )
                
                return result, trace
            
            except Exception as e:
                if attempt == max_retries - 1:
                    error_result = {"success": False, "error": str(e)}
                    trace = DecisionTrace(
                        timestamp=trace_start,
                        state=AgentState.FAILED,
                        reasoning=f"Failed after {max_retries} attempts: {str(e)}",
                        action=f"execute_{step.tool_name}",
                        input_context={"step": step.dict()},
                        output=error_result,
                        duration_ms=(time.time() - start) * 1000
                    )
                    return error_result, trace
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def decision_maker(
        self,
        goal: str,
        context: Dict[str, Any],
        max_steps: int = 10,
        timeout_seconds: int = 30
    ) -> AgentResponse:
        """Main decision-making loop"""
        start_time = time.time()
        traces: List[DecisionTrace] = []
        results = {"steps_completed": [], "final_output": None}
        current_state = AgentState.IDLE
        
        try:
            # Planning phase
            current_state = AgentState.PLANNING
            plan = await asyncio.wait_for(
                self.generate_plan(goal, context),
                timeout=timeout_seconds / 3
            )
            
            traces.append(DecisionTrace(
                timestamp=datetime.now(),
                state=AgentState.PLANNING,
                reasoning=f"Generated plan with {len(plan.steps)} steps",
                action="generate_plan",
                input_context={"goal": goal},
                output={"plan": plan.dict()},
                duration_ms=(time.time() - start_time) * 1000
            ))
            
            if not plan.steps:
                # Generate fallback plan if LLM plan is empty
                logger.warning("empty_plan_fallback", goal=goal)
                plan = self._generate_fallback_plan(goal, context)
                if not plan.steps:
                    raise ValueError("Failed to generate valid plan even with fallback")
            
            # Execution phase
            current_state = AgentState.EXECUTING
            for i, step in enumerate(plan.steps[:max_steps]):
                if time.time() - start_time > timeout_seconds:
                    logger.warning("execution_timeout", goal=goal)
                    break
                
                # Deliberate on step
                deliberation = await self.deliberate_step(step, context)
                
                if not deliberation.get("should_execute", True):
                    traces.append(DecisionTrace(
                        timestamp=datetime.now(),
                        state=AgentState.EVALUATING,
                        reasoning=deliberation.get("reasoning", "Step skipped"),
                        action="skip_step",
                        input_context={"step": step.dict()},
                        duration_ms=0
                    ))
                    continue
                
                # Execute step
                result, trace = await self.execute_step(step, context)
                traces.append(trace)
                
                # Update context
                context["last_result"] = result
                context["step_count"] = i + 1
                results["steps_completed"].append({
                    "step_id": step.step_id,
                    "result": result
                })
                
                if not result.get("success", False):
                    logger.warning("step_failed", step_id=step.step_id, error=result.get("error"))
            
            # Evaluation phase
            current_state = AgentState.EVALUATING
            results["final_output"] = context.get("last_result")
            current_state = AgentState.COMPLETED
            success = True
            
        except asyncio.TimeoutError:
            logger.error("agent_timeout", goal=goal, timeout=timeout_seconds)
            current_state = AgentState.FAILED
            success = False
            results["error"] = "Execution timeout"
        
        except Exception as e:
            logger.error("agent_error", goal=goal, error=str(e))
            current_state = AgentState.FAILED
            success = False
            results["error"] = str(e)
        
        total_duration = (time.time() - start_time) * 1000
        
        return AgentResponse(
            goal=goal,
            state=current_state,
            plan=plan if 'plan' in locals() else None,
            results=results,
            traces=traces,
            total_duration_ms=total_duration,
            success=success
        )

# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ============================================================================
# FastAPI Application
# ============================================================================

tool_registry = ToolRegistry()
decision_maker = DecisionMaker(tool_registry)

# Metrics tracking
metrics_stats = {
    "total_executions": 0,
    "successful_executions": 0,
    "failed_executions": 0,
    "total_steps_executed": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("agent_startup", tools=len(tool_registry.tools))
    yield
    logger.info("agent_shutdown")

app = FastAPI(
    title="L8: Agent Decision Maker",
    description="Autonomous agent with planning, deliberation, and tool orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "lesson": "L8: Core Agent Theory",
        "status": "operational",
        "components": ["decision_maker", "planner", "tool_registry", "state_tracker"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tools_available": len(tool_registry.tools),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/tools")
async def list_tools():
    """List all available tools"""
    tools = tool_registry.list_tools()
    return {
        "tools": [t.dict() for t in tools],
        "count": len(tools)
    }

@app.post("/agent/execute")
async def execute_goal(goal_request: AgentGoal):
    """Execute an agent goal"""
    try:
        metrics_stats["total_executions"] += 1
        response = await decision_maker.decision_maker(
            goal=goal_request.goal,
            context=goal_request.context,
            max_steps=goal_request.max_steps,
            timeout_seconds=goal_request.timeout_seconds
        )
        
        # Update metrics
        if response.success:
            metrics_stats["successful_executions"] += 1
        else:
            metrics_stats["failed_executions"] += 1
        
        if response.results and "steps_completed" in response.results:
            metrics_stats["total_steps_executed"] += len(response.results["steps_completed"])
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "execution_complete",
            "goal": goal_request.goal,
            "success": response.success,
            "duration_ms": response.total_duration_ms
        })
        
        return response
    
    except Exception as e:
        metrics_stats["failed_executions"] += 1
        logger.error("execution_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "tools_registered": len(tool_registry.tools),
        "max_decision_time_seconds": decision_maker.max_decision_time,
        "active_websocket_connections": len(manager.active_connections),
        "total_executions": metrics_stats["total_executions"],
        "successful_executions": metrics_stats["successful_executions"],
        "failed_executions": metrics_stats["failed_executions"],
        "total_steps_executed": metrics_stats["total_steps_executed"],
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
