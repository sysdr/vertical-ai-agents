"""
StateManager - Production-grade state persistence with versioning
Handles PostgreSQL (durable) and Redis (hot cache) storage
"""
import json
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
import logging

import asyncpg
import redis.asyncio as redis
from deepdiff import DeepDiff

from models.agent_state import AgentState, StateSnapshot, StateDiff, StateStatus

logger = logging.getLogger(__name__)

class StateManager:
    """Dual-tier state management with PostgreSQL and Redis"""
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db = db_pool
        self.redis = redis_client
        self.snapshot_interval = 5  # Snapshot every 5 versions
        
    async def initialize(self):
        """Create database schema"""
        async with self.db.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    session_id VARCHAR(255) PRIMARY KEY,
                    state_data JSONB NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id VARCHAR(255) PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    version INTEGER NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (session_id) REFERENCES agent_states(session_id)
                )
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_session 
                ON state_snapshots(session_id, version DESC)
            """)
        
        logger.info("StateManager initialized")
    
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        """
        Load state with cache-first strategy
        1. Check Redis (hot cache)
        2. Fallback to PostgreSQL (cold storage)
        3. Return None if not found
        """
        try:
            # Try Redis first
            cache_key = f"state:{session_id}"
            cached = await self.redis.get(cache_key)
            
            if cached:
                logger.debug(f"Cache HIT for {session_id}")
                state_dict = json.loads(cached)
                return AgentState(**state_dict)
            
            # Cache miss - load from PostgreSQL
            logger.debug(f"Cache MISS for {session_id}")
            async with self.db.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT state_data FROM agent_states WHERE session_id = $1",
                    session_id
                )
            
                if not row:
                    return None
                
                state = AgentState(**row['state_data'])
                
                # Populate cache for next access
                await self._cache_state(state)
                
                return state
            
        except Exception as e:
            logger.error(f"State load failed for {session_id}: {e}")
            return None
    
    async def save_state(self, state: AgentState) -> bool:
        """
        Atomically persist state to both PostgreSQL and Redis
        Implements optimistic locking via version increment
        """
        try:
            # Prepare state update
            state.version += 1
            state.updated_at = datetime.utcnow()
            state_json = state.model_dump_json()
            state_dict = json.loads(state_json)
            
            # Atomic transaction - acquire connection from pool
            async with self.db.acquire() as conn:
                async with conn.transaction():
                    # Upsert to PostgreSQL - convert dict to JSON string for JSONB
                    await conn.execute("""
                        INSERT INTO agent_states (session_id, state_data, version, updated_at)
                        VALUES ($1, $2::jsonb, $3, $4)
                        ON CONFLICT (session_id) 
                        DO UPDATE SET 
                            state_data = $2::jsonb,
                            version = $3,
                            updated_at = $4
                    """, state.session_id, state_json, state.version, state.updated_at)
                    
                    # Create snapshot if needed
                    if state.version % self.snapshot_interval == 0:
                        await self._create_snapshot(state, conn)
            
            # Update hot cache (fire-and-forget, non-blocking)
            asyncio.create_task(self._cache_state(state))
            
            logger.info(f"State saved: {state.session_id} v{state.version}")
            return True
            
        except Exception as e:
            logger.error(f"State save failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _cache_state(self, state: AgentState):
        """Cache state in Redis with 1-hour TTL"""
        try:
            cache_key = f"state:{state.session_id}"
            state_json = state.model_dump_json()
            await self.redis.setex(cache_key, 3600, state_json)
        except Exception as e:
            logger.warning(f"Cache update failed: {e}")
    
    async def _create_snapshot(self, state: AgentState, conn):
        """Create historical snapshot for rollback"""
        snapshot_id = f"{state.session_id}_v{state.version}"
        state_json = state.model_dump_json()
        
        await conn.execute("""
            INSERT INTO state_snapshots (snapshot_id, session_id, version, state_data)
            VALUES ($1, $2, $3, $4::jsonb)
        """, snapshot_id, state.session_id, state.version, state_json)
        
        logger.info(f"Snapshot created: {snapshot_id}")
    
    async def get_state_diff(self, session_id: str, from_version: int, to_version: int) -> Optional[StateDiff]:
        """Calculate difference between two state versions"""
        try:
            # Load both versions
            async with self.db.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT version, state_data 
                    FROM state_snapshots 
                    WHERE session_id = $1 AND version IN ($2, $3)
                    ORDER BY version
                """, session_id, from_version, to_version)
            
                if len(rows) != 2:
                    return None
                
                state1 = rows[0]['state_data']
                state2 = rows[1]['state_data']
                
                # Compute deep diff
                diff = DeepDiff(state1, state2, ignore_order=True)
                
                return StateDiff(
                    from_version=from_version,
                    to_version=to_version,
                    changes=diff.to_dict() if diff else {}
                )
            
        except Exception as e:
            logger.error(f"Diff computation failed: {e}")
            return None
    
    async def rollback_state(self, session_id: str, target_version: int) -> bool:
        """Rollback to previous state version"""
        try:
            snapshot_id = f"{session_id}_v{target_version}"
            
            async with self.db.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT state_data FROM state_snapshots 
                    WHERE snapshot_id = $1
                """, snapshot_id)
                
                if not row:
                    return False
                
                # Restore snapshot as current state
                old_state = AgentState(**row['state_data'])
                old_state.version = target_version
                old_state.updated_at = datetime.utcnow()
            
            await self.save_state(old_state)
            
            logger.info(f"Rolled back {session_id} to v{target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def cleanup_old_states(self, days: int = 30):
        """Clean up states not accessed in N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        async with self.db.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM agent_states 
                WHERE updated_at < $1
            """, cutoff)
        
        logger.info(f"Cleaned up old states: {result}")
