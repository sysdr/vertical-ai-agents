"""
Agent Memory Manager - Dual-Tier System
Short-term: In-memory dict for session context
Long-term: File-based JSON for persistent storage
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class MemoryManager:
    def __init__(self, data_dir: str = "data/users"):
        self.sessions: Dict[str, List[dict]] = {}  # Short-term memory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "short_term_hits": 0,
            "long_term_hits": 0,
            "stores": 0,
            "retrievals": 0
        }
    
    def store_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> dict:
        """Store message in short-term memory"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append(message)
        
        # Keep last 20 messages (working memory limit)
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
        
        self.metrics["stores"] += 1
        return message
    
    def recall_session(self, session_id: str, limit: int = 10) -> List[dict]:
        """Recall recent messages from short-term memory"""
        self.metrics["short_term_hits"] += 1
        self.metrics["retrievals"] += 1
        
        if session_id in self.sessions:
            return self.sessions[session_id][-limit:]
        return []
    
    def save_to_longterm(self, user_id: str, fact: dict) -> bool:
        """Persist fact to long-term storage"""
        try:
            filepath = self.data_dir / f"{user_id}.json"
            
            # Load existing data
            existing = []
            if filepath.exists():
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            
            # Append new fact
            entry = {
                "timestamp": datetime.now().isoformat(),
                "fact": fact
            }
            existing.append(entry)
            
            # Save with pretty printing
            with open(filepath, 'w') as f:
                json.dump(existing, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Long-term storage error: {e}")
            return False
    
    def search_longterm(self, user_id: str, keywords: List[str] = None, limit: int = 5) -> List[dict]:
        """Search long-term memory for relevant facts"""
        self.metrics["long_term_hits"] += 1
        self.metrics["retrievals"] += 1
        
        try:
            filepath = self.data_dir / f"{user_id}.json"
            
            if not filepath.exists():
                return []
            
            with open(filepath, 'r') as f:
                entries = json.load(f)
            
            # If no keywords, return recent entries
            if not keywords:
                return sorted(entries, key=lambda x: x['timestamp'], reverse=True)[:limit]
            
            # Simple keyword matching
            matches = []
            for entry in entries:
                fact_text = json.dumps(entry['fact']).lower()
                if any(kw.lower() in fact_text for kw in keywords):
                    matches.append(entry)
            
            # Sort by recency
            matches = sorted(matches, key=lambda x: x['timestamp'], reverse=True)[:limit]
            return matches
            
        except Exception as e:
            print(f"Long-term search error: {e}")
            return []
    
    def get_metrics(self) -> dict:
        """Return memory system metrics"""
        return {
            **self.metrics,
            "active_sessions": len(self.sessions),
            "total_session_messages": sum(len(msgs) for msgs in self.sessions.values())
        }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear short-term memory for session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
