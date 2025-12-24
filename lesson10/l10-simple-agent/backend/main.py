from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from agent import SimpleAgent

app = FastAPI(title="L10 Simple Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent - get API key from environment variable
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your .env file or environment.")
agent = SimpleAgent(api_key=GEMINI_KEY)

class Message(BaseModel):
    content: str
    goal: Optional[str] = None

@app.get("/")
def root():
    return {"message": "L10 Simple Agent API", "status": "running"}

@app.get("/agent/state")
def get_agent_state():
    return agent.get_state()

@app.post("/agent/act")
async def agent_act(message: Message):
    try:
        result = await agent.act(message.content, message.goal)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/memory")
def get_memory():
    return {
        "short_term": {
            "conversation_history": agent.short_term.conversation_history,
            "memory_keys": list(agent.short_term.memory.keys())
        },
        "long_term_count": len(agent.long_term.data),
        "decision_log_count": len(agent.decision_log.data)
    }

@app.get("/agent/decisions")
def get_decisions():
    return {
        "decisions": list(agent.decision_log.data.values())[-10:]  # Last 10 decisions
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
