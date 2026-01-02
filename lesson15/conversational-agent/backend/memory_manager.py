import aiosqlite
import json
from typing import Optional, List
from models import ConversationStateModel
from datetime import datetime
import os

class MemoryManager:
    def __init__(self, db_path: str = "../data/conversations.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT UNIQUE NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON conversations(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversations(conversation_id)")
            await db.commit()
    
    async def save_state(self, state: ConversationStateModel):
        async with aiosqlite.connect(self.db_path) as db:
            state_json = json.dumps(state.to_dict())
            await db.execute("""
                INSERT OR REPLACE INTO conversations 
                (user_id, conversation_id, state_json, created_at, updated_at, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (state.user_id, state.conversation_id, state_json,
                  state.created_at.isoformat(), state.updated_at.isoformat(), state.total_tokens))
            await db.commit()
    
    async def load_state(self, conversation_id: str) -> Optional[ConversationStateModel]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT state_json FROM conversations WHERE conversation_id = ?", 
                                (conversation_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return ConversationStateModel.from_dict(json.loads(row[0]))
                return None
    
    async def list_conversations(self, user_id: str) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT conversation_id, created_at, updated_at, total_tokens
                FROM conversations WHERE user_id = ? ORDER BY updated_at DESC
            """, (user_id,)) as cursor:
                rows = await cursor.fetchall()
                return [{"conversation_id": r[0], "created_at": r[1], 
                        "updated_at": r[2], "total_tokens": r[3]} for r in rows]
