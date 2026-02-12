from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import json
import os

from validator_agent import ValidatorAgent, Document, ValidatorResponse

app = FastAPI(title="L39: Validator Agent - Risk & Compliance")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global validator instance
validator: Optional[ValidatorAgent] = None

class ValidationRequest(BaseModel):
    documents: List[Dict]
    query: str
    domain: str = "healthcare"
    context: Dict = {}

class ComplianceStats(BaseModel):
    total_validations: int
    total_violations: int
    approval_rate: float
    avg_risk_score: float
    violations_by_rule: Dict[str, int]

@app.on_event("startup")
async def startup_event():
    global validator
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    validator = ValidatorAgent(domain="healthcare", redis_url=redis_url)
    await validator.initialize()
    print("✅ ValidatorAgent initialized")

@app.post("/validate", response_model=ValidatorResponse)
async def validate_content(request: ValidationRequest):
    """Validate retrieved documents for factual consistency and compliance"""
    
    if not validator:
        raise HTTPException(status_code=500, detail="Validator not initialized")
    
    # Convert dict documents to Document objects
    documents = [
        Document(
            id=doc.get('id', f"doc_{i}"),
            content=doc.get('content', ''),
            source=doc.get('source', 'unknown'),
            metadata=doc.get('metadata', {})
        )
        for i, doc in enumerate(request.documents)
    ]
    
    result = await validator.validate(
        documents=documents,
        query=request.query,
        context=request.context
    )
    
    return result

@app.get("/stats", response_model=ComplianceStats)
async def get_statistics():
    """Get compliance validation statistics"""
    
    if not validator:
        raise HTTPException(status_code=500, detail="Validator not initialized")
    
    stats = await validator.get_stats()
    return ComplianceStats(**stats)

@app.get("/audit-trail")
async def get_audit_trail(limit: int = 50):
    """Get recent audit trail entries"""
    
    if not validator:
        raise HTTPException(status_code=500, detail="Validator not initialized")
    
    trail = await validator.get_audit_trail(limit)
    return {"audit_trail": trail}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "validator_initialized": validator is not None
    }

# WebSocket for real-time updates
connected_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        while True:
            # Send periodic stats updates
            if validator:
                stats = await validator.get_stats()
                await websocket.send_json({
                    "type": "stats_update",
                    "data": stats,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
