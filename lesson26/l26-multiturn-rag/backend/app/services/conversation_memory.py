import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Manages conversation history with configurable window strategies"""
    
    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: int = 2000,
        strategy: str = "sliding"  # sliding, token-based, semantic
    ):
        self.buffer: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def add_message(self, role: str, content: str):
        """Add message to buffer and apply window strategy"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.buffer.append(message)
        self._apply_window_strategy()
    
    def _apply_window_strategy(self):
        """Apply configured memory window strategy"""
        if self.strategy == "sliding":
            self._apply_sliding_window()
        elif self.strategy == "token-based":
            self._apply_token_based_pruning()
        else:
            self._apply_sliding_window()
    
    def _apply_sliding_window(self):
        """Keep last N messages"""
        if len(self.buffer) > self.max_messages:
            self.buffer = self.buffer[-self.max_messages:]
            logger.debug(f"Applied sliding window, kept {len(self.buffer)} messages")
    
    def _apply_token_based_pruning(self):
        """Prune based on token count"""
        total_tokens = self._count_tokens()
        
        while total_tokens > self.max_tokens and len(self.buffer) > 2:
            removed = self.buffer.pop(0)
            total_tokens = self._count_tokens()
            logger.debug(f"Pruned message, now {total_tokens} tokens")
        
        # Also apply message limit
        if len(self.buffer) > self.max_messages:
            self.buffer = self.buffer[-self.max_messages:]
    
    def _count_tokens(self) -> int:
        """Count total tokens in buffer"""
        total = 0
        for msg in self.buffer:
            tokens = self.tokenizer.encode(msg["content"])
            total += len(tokens)
        return total
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation messages"""
        if last_n:
            return self.buffer[-last_n:]
        return self.buffer.copy()
    
    def get_context_string(self, last_n: Optional[int] = None) -> str:
        """Get formatted conversation context"""
        messages = self.get_messages(last_n)
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
    
    def clear(self):
        """Clear conversation buffer"""
        self.buffer = []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "message_count": len(self.buffer),
            "total_tokens": self._count_tokens(),
            "strategy": self.strategy,
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "buffer": self.buffer,
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
            "strategy": self.strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Deserialize from dict"""
        memory = cls(
            max_messages=data.get("max_messages", 10),
            max_tokens=data.get("max_tokens", 2000),
            strategy=data.get("strategy", "sliding")
        )
        memory.buffer = data.get("buffer", [])
        return memory
