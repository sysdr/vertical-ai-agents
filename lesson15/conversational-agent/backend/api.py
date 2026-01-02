from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from conversation_engine import ConversationEngine
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Conversational Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = os.getenv("DATABASE_PATH", "../data/conversations.db")

if not API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment or .env file")
    print("   Please set GEMINI_API_KEY in conversational-agent/backend/.env")
    print("   Get a key from: https://aistudio.google.com/app/apikey")

engine = ConversationEngine(API_KEY, DB_PATH)

class CreateConversationRequest(BaseModel):
    user_id: str

class MessageRequest(BaseModel):
    conversation_id: str
    message: str

@app.on_event("startup")
async def startup():
    await engine.initialize()

@app.post("/conversations")
async def create_conversation(req: CreateConversationRequest):
    conversation_id = await engine.create_conversation(req.user_id)
    return {"conversation_id": conversation_id}

@app.post("/messages")
async def send_message(req: MessageRequest):
    try:
        result = await engine.process_message(req.conversation_id, req.message)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        # Check if it's an API key error
        error_msg = str(e)
        if "API key" in error_msg or "API_KEY" in error_msg or "expired" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="API key configuration issue. Please update the GEMINI_API_KEY in backend/.env and restart the service."
            )
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/conversations/{conversation_id}/history")
async def get_history(conversation_id: str):
    state = await engine.memory.load_state(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "messages": [m.dict() for m in state.messages],
        "goals": [g.dict() for g in state.active_goals],
        "state": state.state.value
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
