import redis.asyncio as redis
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.session_ttl = int(os.getenv("SESSION_TTL_HOURS", "24"))
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = await redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True
            )
            await self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Fallback to in-memory dict for development
            self.client = None
            logger.warning("Using in-memory storage as fallback")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
    
    async def health_check(self) -> bool:
        """Check Redis health"""
        if not self.client:
            return False
        try:
            await self.client.ping()
            return True
        except:
            return False
    
    async def save_conversation(self, session_id: str, conversation: Dict[str, Any]):
        """Save conversation history"""
        if not self.client:
            return
        
        key = f"conversation:{session_id}"
        data = json.dumps(conversation, default=str)
        ttl = timedelta(hours=self.session_ttl)
        
        await self.client.setex(key, ttl, data)
        logger.debug(f"Saved conversation {session_id}")
    
    async def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation history"""
        if not self.client:
            return None
        
        key = f"conversation:{session_id}"
        data = await self.client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def delete_conversation(self, session_id: str):
        """Delete conversation"""
        if not self.client:
            return
        
        key = f"conversation:{session_id}"
        await self.client.delete(key)
        logger.info(f"Deleted conversation {session_id}")
    
    async def list_sessions(self) -> list:
        """List all active sessions"""
        if not self.client:
            return []
        
        keys = await self.client.keys("conversation:*")
        return [k.replace("conversation:", "") for k in keys]

redis_service = RedisService()
