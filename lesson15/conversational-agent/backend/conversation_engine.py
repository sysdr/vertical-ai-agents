from models import ConversationStateModel, ConversationState
from memory_manager import MemoryManager
from goal_tracker import GoalTracker
from gemini_client import GeminiClient
from typing import Optional
import uuid

class ConversationEngine:
    def __init__(self, api_key: str, db_path: str = "../data/conversations.db"):
        self.memory = MemoryManager(db_path)
        self.goal_tracker = GoalTracker(api_key)
        self.llm = GeminiClient(api_key)
    
    async def initialize(self):
        await self.memory.initialize()
    
    async def create_conversation(self, user_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        state = ConversationStateModel(user_id=user_id, conversation_id=conversation_id)
        await self.memory.save_state(state)
        return conversation_id
    
    async def process_message(self, conversation_id: str, user_message: str) -> dict:
        state = await self.memory.load_state(conversation_id)
        if not state:
            return {"error": "Conversation not found"}
        
        # Add user message
        state.add_message("user", user_message)
        
        # Check for goal commands
        if user_message.lower().startswith("/goal "):
            goal_text = user_message[6:].strip()
            state.add_goal(goal_text)
            response_text = f"Goal set: {goal_text}"
            state.add_message("assistant", response_text)
        else:
            # Build context with goals
            goals_context = self.goal_tracker.get_active_goals_context(state)
            system_context = f"""You are a helpful AI assistant with persistent memory.
You remember previous conversations and work towards user goals.
{goals_context}
Respond naturally and helpfully."""
            
            # Generate response
            response_text, tokens = await self.llm.generate_response(state.messages, system_context)
            state.add_message("assistant", response_text, tokens)
            
            # Update goals
            await self.goal_tracker.update_goals(state)
        
        # Save state
        await self.memory.save_state(state)
        
        return {
            "response": response_text,
            "state": state.state.value,
            "active_goals": len([g for g in state.active_goals if not g.completed]),
            "total_messages": len(state.messages),
            "total_tokens": state.total_tokens
        }
