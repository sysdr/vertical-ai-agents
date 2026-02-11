import redis.asyncio as redis
import json
import logging
from typing import Optional
from config.settings import settings
from models.document import ValidationScore

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}. Running without cache.")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_validation(self, pair_hash: str) -> Optional[ValidationScore]:
        """Get cached validation score"""
        if not self.redis_client:
            return None
            
        try:
            key = f"validation:{pair_hash}"
            data = await self.redis_client.get(key)
            if data:
                return ValidationScore.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set_validation(self, pair_hash: str, score: ValidationScore):
        """Cache validation score"""
        if not self.redis_client:
            return
            
        try:
            key = f"validation:{pair_hash}"
            await self.redis_client.setex(
                key,
                settings.CACHE_TTL,
                score.model_dump_json()
            )
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

cache_service = CacheService()
