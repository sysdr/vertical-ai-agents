from fastapi import APIRouter, HTTPException
from typing import List
import uuid
import logging
from datetime import datetime

from app.models.schemas import (
    QueryRequest, QueryResponse, ConversationHistory,
    SessionInfo, HealthResponse, Message
)
from app.services.redis_service import redis_service
from app.services.rag_service import rag_service
from app.services.conversation_memory import ConversationMemory

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process query with multi-turn context"""
    try:
        # Load or create conversation
        conv_data = await redis_service.get_conversation(request.session_id)
        
        if conv_data:
            memory = ConversationMemory.from_dict(conv_data["memory"])
            state = conv_data.get("state", "ACTIVE")
        else:
            memory = ConversationMemory()
            state = "NEW"
            conv_data = {
                "session_id": request.session_id,
                "created_at": datetime.now().isoformat(),
                "memory": memory.to_dict(),
                "state": state
            }
        
        # Add user query to memory
        memory.add_message("user", request.query)
        
        # Process query
        response, is_ambiguous, ambiguity_score, clarifications, documents = \
            await rag_service.query(request.query, memory)
        
        # Update state
        if is_ambiguous:
            state = "CLARIFYING"
        else:
            state = "ACTIVE"
            # Add assistant response to memory
            memory.add_message("assistant", response)
        
        # Save conversation
        conv_data["memory"] = memory.to_dict()
        conv_data["state"] = state
        conv_data["updated_at"] = datetime.now().isoformat()
        await redis_service.save_conversation(request.session_id, conv_data)
        
        return QueryResponse(
            session_id=request.session_id,
            response=response if response else "Please clarify your question:",
            is_ambiguous=is_ambiguous,
            ambiguity_score=ambiguity_score,
            clarification_questions=clarifications,
            retrieved_documents=documents,
            context_used=[
                Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
                for msg in memory.get_messages()
            ],
            conversation_state=state
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Get conversation history"""
    conv_data = await redis_service.get_conversation(session_id)
    
    if not conv_data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    memory = ConversationMemory.from_dict(conv_data["memory"])
    
    return ConversationHistory(
        session_id=session_id,
        messages=[
            Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
            for msg in memory.get_messages()
        ],
        created_at=conv_data["created_at"],
        updated_at=conv_data.get("updated_at", conv_data["created_at"])
    )

@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """Delete conversation"""
    await redis_service.delete_conversation(session_id)
    return {"message": "Conversation deleted"}

@router.get("/sessions", response_model=List[str])
async def list_sessions():
    """List active sessions"""
    sessions = await redis_service.list_sessions()
    return sessions

@router.get("/session/{session_id}/info", response_model=SessionInfo)
async def session_info(session_id: str):
    """Get session information. Returns default info for new sessions not yet stored."""
    conv_data = await redis_service.get_conversation(session_id)
    
    if not conv_data:
        # Session created via POST /session/new but no query yet — return default info
        now = datetime.now()
        return SessionInfo(
            session_id=session_id,
            message_count=0,
            total_tokens=0,
            created_at=now,
            last_activity=now,
            state="NEW"
        )
    
    memory = ConversationMemory.from_dict(conv_data["memory"])
    metadata = memory.get_metadata()
    
    return SessionInfo(
        session_id=session_id,
        message_count=metadata["message_count"],
        total_tokens=metadata["total_tokens"],
        created_at=conv_data["created_at"],
        last_activity=conv_data.get("updated_at", conv_data["created_at"]),
        state=conv_data.get("state", "UNKNOWN")
    )

@router.post("/session/new")
async def create_session():
    """Create new session"""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}
