from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path
import uvicorn
from app.budget_manager import BudgetManager, BudgetException
from app.execution_controller import BudgetedReflexionAgent
from app.gemini_client import GeminiClient
from datetime import datetime
import json

# Config path: works when run from backend/ or from /app (Docker)
_BASE = Path(__file__).resolve().parent
CONFIG_DIR = _BASE.parent / "config" if (_BASE.parent / "config").exists() else _BASE / "config"

app = FastAPI(title="L34: Planning Loop Controls")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics storage
metrics_store = []

class AgentRequest(BaseModel):
    task: str
    environment: str = "production"
    max_iterations: Optional[int] = None
    max_tokens: Optional[int] = None

class AgentResponse(BaseModel):
    success: bool
    result: Optional[str]
    error: Optional[str]
    metrics: Dict
    timestamp: str

@app.post("/api/agent/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute budgeted ReAct agent with iteration and token limits."""
    try:
        # Load environment-specific budget policy
        with open(CONFIG_DIR / f"{request.environment}_policy.json", "r") as f:
            policy = json.load(f)
        
        # Override with custom limits if provided
        if request.max_iterations:
            policy['max_iterations'] = request.max_iterations
        if request.max_tokens:
            policy['max_tokens_per_turn'] = request.max_tokens
        
        # Initialize agent
        gemini_client = GeminiClient()
        agent = BudgetedReflexionAgent(gemini_client, policy)
        
        # Execute task
        result = await agent.run(request.task)
        
        # Get usage metrics
        metrics = agent.get_metrics()
        metrics_store.append({
            'timestamp': datetime.now().isoformat(),
            'task': request.task[:50],
            'environment': request.environment,
            **metrics
        })
        
        return AgentResponse(
            success=True,
            result=result,
            error=None,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
    
    except BudgetException as e:
        # Handle budget violations gracefully
        partial_metrics = agent.get_metrics() if 'agent' in locals() else {}
        metrics_store.append({
            'timestamp': datetime.now().isoformat(),
            'task': request.task[:50],
            'environment': request.environment,
            'error': str(e),
            **partial_metrics
        })
        
        return AgentResponse(
            success=False,
            result=None,
            error=str(e),
            metrics=partial_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/recent")
async def get_recent_metrics(limit: int = 20):
    """Retrieve recent execution metrics."""
    return {
        'metrics': metrics_store[-limit:],
        'total_executions': len(metrics_store)
    }

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get aggregated metrics summary."""
    if not metrics_store:
        return {'summary': 'No executions yet'}
    
    total_iterations = sum(m.get('total_iterations', 0) for m in metrics_store)
    total_tokens = sum(m.get('total_tokens', 0) for m in metrics_store)
    budget_violations = sum(1 for m in metrics_store if 'error' in m)
    
    return {
        'total_executions': len(metrics_store),
        'average_iterations': total_iterations / len(metrics_store) if metrics_store else 0,
        'average_tokens': total_tokens / len(metrics_store) if metrics_store else 0,
        'budget_violations': budget_violations,
        'violation_rate': budget_violations / len(metrics_store) if metrics_store else 0
    }

@app.get("/api/policies")
async def get_policies():
    """Retrieve available budget policies."""
    policies = {}
    for env in ['development', 'staging', 'production']:
        try:
            with open(CONFIG_DIR / f"{env}_policy.json", "r") as f:
                policies[env] = json.load(f)
        except FileNotFoundError:
            continue
    return policies

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "L34 Planning Loop Controls"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
