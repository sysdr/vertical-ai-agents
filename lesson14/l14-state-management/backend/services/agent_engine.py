"""
Agent Engine - Integrates Gemini AI with state management
"""
import google.generativeai as genai
from typing import Optional
import logging

from models.agent_state import AgentState, ConversationTurn, StateStatus
from services.state_manager import StateManager

logger = logging.getLogger(__name__)

class AgentEngine:
    """Agent processing engine with stateful conversation"""
    
    def __init__(self, state_manager: StateManager, api_key: str):
        self.state_manager = state_manager
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required but not provided")
        genai.configure(api_key=api_key)
        # Try gemini-1.5-flash first (most stable), fallback to gemini-pro
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logger.warning(f"Failed to initialize gemini-1.5-flash: {e}, trying gemini-pro")
            self.model = genai.GenerativeModel('gemini-pro')
    
    async def process_message(self, session_id: str, user_message: str) -> dict:
        """
        Process user message with state continuity
        Returns: {response, state_version, tokens_used}
        """
        try:
            # Load current state
            state = await self.state_manager.load_state(session_id)
            
            if not state:
                # Initialize new session
                state = AgentState(session_id=session_id)
            
            # Update state to processing
            state.state_status = StateStatus.PROCESSING
            
            # Build context from conversation history
            context = self._build_context(state)
            
            # Generate response with Gemini
            prompt = f"{context}\n\nUser: {user_message}\nAgent:"
            response = await self._generate_response(prompt, state)
            
            # Update conversation history
            turn = ConversationTurn(
                turn_id=state.total_turns + 1,
                user_message=user_message,
                agent_response=response['text'],
                tokens_used=response['tokens']
            )
            
            state.conversation_history.append(turn)
            state.total_turns += 1
            state.total_tokens += response['tokens']
            state.state_status = StateStatus.AWAITING_INPUT
            
            # Persist updated state
            await self.state_manager.save_state(state)
            
            return {
                'response': response['text'],
                'state_version': state.version,
                'tokens_used': response['tokens'],
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            if state:
                state.state_status = StateStatus.FAILED
                await self.state_manager.save_state(state)
            raise
    
    def _build_context(self, state: AgentState) -> str:
        """Build conversation context from state"""
        if not state.conversation_history:
            return "You are a helpful AI assistant."
        
        # Use last 5 turns for context
        recent_turns = state.conversation_history[-5:]
        context_parts = ["Previous conversation:"]
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Agent: {turn.agent_response}")
        
        return "\n".join(context_parts)
    
    async def _generate_response(self, prompt: str, state: AgentState) -> dict:
        """Generate response with Gemini"""
        try:
            # Run the synchronous API call in a thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini API")
            
            # Estimate tokens (rough approximation)
            tokens = len(prompt.split()) + len(response.text.split())
            
            return {
                'text': response.text,
                'tokens': tokens
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini API error: {error_msg}", exc_info=True)
            
            # Check if it's an API key error and provide fallback demo response
            if "API key" in error_msg.lower() or "API_KEY" in error_msg or "expired" in error_msg.lower():
                # Provide fallback demo response for development/testing
                demo_response = self._get_demo_response(prompt, state)
                logger.warning("Using demo fallback response due to API key error")
                return {
                    'text': demo_response,
                    'tokens': len(prompt.split()) + len(demo_response.split())
                }
            elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                error_text = "API quota exceeded: Please check your Gemini API usage limits."
                return {
                    'text': error_text,
                    'tokens': 0
                }
            else:
                error_text = f"I encountered an error: {error_msg[:200]}"
            return {
                    'text': error_text,
                'tokens': 0
                }
    
    def _get_demo_response(self, prompt: str, state: AgentState) -> str:
        """Generate a demo response when API key is invalid (for development/testing)"""
        prompt_lower = prompt.lower()
        user_message = prompt.split('\n\nUser: ')[-1].split('\nAgent:')[0].strip() if 'User: ' in prompt else prompt
        
        # Check if asking for metrics/statistics
        metrics_patterns = [
            'display memory metrics', 'show memory metrics', 'memory metrics', 'metrics',
            'show metrics', 'display metrics', 'statistics', 'stats', 'show stats',
            'session stats', 'conversation stats', 'memory stats', 'performance metrics'
        ]
        is_metrics_request = any(pattern in prompt_lower for pattern in metrics_patterns)
        
        # Check if user is asking about memory/conversation history
        # Use more specific patterns to avoid false positives
        memory_patterns = [
            'what do you remember', 'what do you recall', 'what did we', 'what did you',
            'do you remember', 'do you recall', 'can you remember', 'can you recall',
            'conversation history', 'session history', 'what we talked', 'what we discussed',
            'what i said', 'what i told', 'previous message', 'earlier message'
        ]
        is_memory_question = any(pattern in prompt_lower for pattern in memory_patterns)
        
        # Check if asking about system architecture/technology
        tech_keywords = {
            'redis': ['redis', 'cache', 'caching', 'hot cache', 'in-memory'],
            'postgresql': ['postgres', 'postgresql', 'database', 'db', 'persistence', 'storage'],
            'architecture': ['architecture', 'design', 'system design', 'how does it work', 'how it works'],
            'state': ['state management', 'state persistence', 'state versioning', 'state snapshots'],
            'fastapi': ['fastapi', 'api', 'endpoint', 'backend'],
            'dual-tier': ['dual-tier', 'two-tier', 'hot cold', 'hot/cold']
        }
        
        # Check for technical questions
        is_tech_question = False
        tech_topic = None
        for topic, keywords in tech_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                is_tech_question = True
                tech_topic = topic
                break
        
        # If asking about system architecture/technology, provide informative answers
        if is_tech_question:
            tech_responses = {
                'redis': """Redis was chosen for caching in this system for several key reasons:

1. **Performance**: Redis is an in-memory data store, providing sub-millisecond latency for state retrieval. This makes it ideal for a "hot cache" layer.

2. **Dual-Tier Architecture**: The system uses a dual-tier approach:
   - **Redis (Hot Cache)**: Fast, in-memory storage for frequently accessed states (1-hour TTL)
   - **PostgreSQL (Cold Storage)**: Durable, persistent storage for long-term state retention

3. **Cache-First Strategy**: When loading state, the system:
   - First checks Redis (cache hit = instant response)
   - Falls back to PostgreSQL if cache miss
   - Automatically populates cache for next access

4. **Scalability**: Redis handles high read throughput, reducing load on PostgreSQL
5. **TTL Support**: Built-in expiration (1 hour) ensures cache freshness
6. **Async Support**: aioredis provides excellent async/await support for FastAPI

This architecture provides the best of both worlds: speed for active sessions and durability for long-term storage.

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)""",
                
                'postgresql': """PostgreSQL serves as the durable, persistent storage layer in this dual-tier architecture:

**Why PostgreSQL:**
1. **Durability**: ACID compliance ensures state is never lost
2. **JSONB Support**: Native JSON storage for flexible state schemas
3. **Reliability**: Proven production database for critical data
4. **Versioning**: Supports state versioning and snapshots for rollback
5. **Cold Storage**: Long-term persistence when Redis cache expires

**Architecture:**
- Redis (hot cache) → Fast access for active sessions
- PostgreSQL (cold storage) → Permanent record of all states
- Automatic fallback: Cache miss → Load from PostgreSQL → Populate cache

This ensures both performance and data integrity.

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)""",
                
                'architecture': """This system uses a **dual-tier state management architecture**:

**Hot Cache (Redis):**
- In-memory storage for active sessions
- 1-hour TTL for automatic expiration
- Sub-millisecond access times
- Cache-first lookup strategy

**Cold Storage (PostgreSQL):**
- Durable, persistent storage
- JSONB for flexible state schemas
- Versioning and snapshots for rollback
- Long-term data retention

**Flow:**
1. State load: Check Redis → If miss, load from PostgreSQL → Cache result
2. State save: Write to PostgreSQL (atomic) → Update Redis cache (async)
3. Snapshots: Created every 5 versions for rollback capability

**Benefits:**
- Fast response times for active sessions
- Data durability and persistence
- Automatic cache management
- Production-grade reliability

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)""",
                
                'state': """The state management system provides:

**Features:**
- **Versioning**: Automatic version increment on each state update
- **Snapshots**: Created every 5 versions for rollback capability
- **Dual Storage**: Redis (hot) + PostgreSQL (cold)
- **State Diff**: Compare any two state versions
- **Rollback**: Restore to any previous snapshot

**State Model Includes:**
- Conversation history (all turns)
- User context and metadata
- Goals and planning
- Statistics (turns, tokens)
- Status tracking (IDLE, PROCESSING, etc.)

**Persistence:**
- Redis: Fast access, 1-hour TTL
- PostgreSQL: Permanent storage
- Automatic cache population

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)""",
                
                'fastapi': """FastAPI powers the backend API with:

**Features:**
- Async/await support for high concurrency
- Automatic API documentation (Swagger/OpenAPI)
- Type validation with Pydantic models
- CORS support for frontend integration

**Endpoints:**
- `POST /api/chat` - Process messages with state
- `GET /api/state/{session_id}` - Retrieve state
- `POST /api/state/diff` - Compare versions
- `POST /api/state/rollback` - Rollback state
- `GET /health` - Health check

Built with production-grade error handling and logging.

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)""",
                
                'dual-tier': """The dual-tier architecture combines speed and durability:

**Tier 1: Redis (Hot Cache)**
- Purpose: Fast, in-memory access
- TTL: 1 hour
- Use case: Active session state
- Performance: Sub-millisecond latency

**Tier 2: PostgreSQL (Cold Storage)**
- Purpose: Durable, persistent storage
- Use case: Long-term state retention
- Features: ACID compliance, JSONB support
- Reliability: Never lose data

**Benefits:**
- Best performance for active sessions
- Guaranteed data persistence
- Automatic cache management
- Production-ready scalability

(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)"""
            }
            
            if tech_topic in tech_responses:
                return tech_responses[tech_topic]
            else:
                return f"That's a great technical question about {tech_topic}! For detailed answers, set up your API key with `./setup_api_key.sh` to enable full AI responses."
        
        # If asking for metrics, display comprehensive statistics
        if is_metrics_request:
            from datetime import datetime
            
            metrics = []
            metrics.append("📊 **Memory Metrics Dashboard**")
            metrics.append("=" * 50)
            
            # Session Information
            metrics.append(f"\n**Session Info:**")
            metrics.append(f"  • Session ID: {state.session_id}")
            metrics.append(f"  • State Version: {state.version}")
            metrics.append(f"  • Status: {state.state_status.value}")
            
            # Timestamps
            if state.created_at:
                created = state.created_at if isinstance(state.created_at, datetime) else datetime.fromisoformat(str(state.created_at).replace('Z', '+00:00'))
                metrics.append(f"  • Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
            if state.last_activity:
                last_act = state.last_activity if isinstance(state.last_activity, datetime) else datetime.fromisoformat(str(state.last_activity).replace('Z', '+00:00'))
                metrics.append(f"  • Last Activity: {last_act.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Conversation Statistics
            metrics.append(f"\n**Conversation Stats:**")
            metrics.append(f"  • Total Turns: {state.total_turns}")
            metrics.append(f"  • Total Tokens: {state.total_tokens:,}")
            if state.total_turns > 0:
                avg_tokens = state.total_tokens / state.total_turns
                metrics.append(f"  • Avg Tokens/Turn: {avg_tokens:.1f}")
            else:
                metrics.append(f"  • Avg Tokens/Turn: 0")
            
            # Conversation History Summary
            if state.conversation_history:
                metrics.append(f"\n**Recent Conversation:**")
                recent = state.conversation_history[-3:]  # Last 3 turns
                for turn in recent:
                    user_preview = turn.user_message[:50] + "..." if len(turn.user_message) > 50 else turn.user_message
                    metrics.append(f"  • Turn {turn.turn_id}: '{user_preview}' ({turn.tokens_used} tokens)")
                if len(state.conversation_history) > 3:
                    metrics.append(f"  ... and {len(state.conversation_history) - 3} more turns")
            else:
                metrics.append(f"\n**Conversation History:** Empty (new session)")
            
            # Goals and Context
            if state.goals:
                metrics.append(f"\n**Goals:** {len(state.goals)} active goal(s)")
            if state.user_context:
                metrics.append(f"\n**User Context:** {len(state.user_context)} context entries")
            
            # Storage Info
            snapshot_interval = 5  # Default snapshot interval
            metrics.append(f"\n**Storage:**")
            metrics.append(f"  • Redis Cache: Active (1-hour TTL)")
            metrics.append(f"  • PostgreSQL: Persistent storage")
            metrics.append(f"  • Snapshots: Every {snapshot_interval} versions")
            next_snapshot = ((state.version // snapshot_interval) + 1) * snapshot_interval
            metrics.append(f"  • Next Snapshot: At version {next_snapshot}")
            
            metrics.append("\n" + "=" * 50)
            metrics.append("(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)")
            
            return "\n".join(metrics)
        
        # If asking about memory
        if is_memory_question:
            if state.conversation_history:
                history_summary = []
                for i, turn in enumerate(state.conversation_history, 1):
                    user_msg = turn.user_message[:80] + "..." if len(turn.user_message) > 80 else turn.user_message
                    agent_msg = turn.agent_response[:80] + "..." if len(turn.agent_response) > 80 else turn.agent_response
                    history_summary.append(f"• Turn {i}: You: '{user_msg}' | Me: '{agent_msg}'")
                
                memory_response = f"I remember our conversation! Here's what we've discussed in this session:\n\n"
                memory_response += "\n".join(history_summary[-5:])  # Show last 5 turns
                if len(state.conversation_history) > 5:
                    memory_response += f"\n\n(Showing last 5 of {state.total_turns} total turns)"
                memory_response += f"\n\nSession stats: {state.total_turns} turns, {state.total_tokens} tokens used"
                memory_response += "\n\n(Note: Demo mode - run `./setup_api_key.sh` for full AI capabilities.)"
                return memory_response
            else:
                return "This is the start of our conversation, so I don't have any previous messages to remember yet. (Demo mode - set up your API key with `./setup_api_key.sh` for full AI capabilities.)"
        
        # More natural, conversational demo responses
        responses = {
            'greeting': [
                "I'm doing well, thank you for asking! 😊 (Note: This is a demo response. Run `./setup_api_key.sh` to enable real AI responses.)",
                "I'm here and ready to help! (Demo mode - get your free API key at https://aistudio.google.com/apikey)",
                "Hello! I'm working, but I need a Gemini API key for full functionality. Run `./setup_api_key.sh` to set it up!"
            ],
            'how_are_you': [
                "I'm doing great, thanks! To get real AI responses, please set up your API key with `./setup_api_key.sh`",
                "I'm fine, thank you! (Currently in demo mode - configure your Gemini API key for full AI capabilities)",
                "Doing well! 😊 For real AI responses, get a free API key: https://aistudio.google.com/apikey"
            ],
            'help': [
                "I'd love to help! To enable full AI functionality, run `./setup_api_key.sh` in the project directory. It's quick and free!",
                "Sure! First, let's get your API key set up. Run `./setup_api_key.sh` - it will guide you through the process.",
                "I can help! To get started, you'll need a Gemini API key. Run `./setup_api_key.sh` for an automated setup."
            ],
            'question': [
                "That's a great question! To get real AI-powered answers, please set up your Gemini API key. Run `./setup_api_key.sh` for easy setup.",
                "I'd love to answer that! First, configure your API key with `./setup_api_key.sh` to enable full AI responses.",
                "Interesting! For detailed AI responses, get your free API key at https://aistudio.google.com/apikey or run `./setup_api_key.sh`"
            ],
            'default': [
                "I understand! To enable real AI responses, please set up your Gemini API key. Run `./setup_api_key.sh` for automated setup.",
                "Got it! For full AI functionality, configure your API key. Visit https://aistudio.google.com/apikey or run `./setup_api_key.sh`",
                "Thanks for your message! To get AI responses, set up your API key with `./setup_api_key.sh` - it's free and takes just a minute."
            ]
        }
        
        import random
        
        # Determine response type
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            return random.choice(responses['greeting'])
        elif any(phrase in prompt_lower for phrase in ['how are you', 'how r u', 'how are u', "how's it going", "how's things"]):
            return random.choice(responses['how_are_you'])
        elif any(word in prompt_lower for word in ['help', 'assist', 'support', 'guide']):
            return random.choice(responses['help'])
        elif any(word in prompt_lower for word in ['what', 'who', 'where', 'when', 'why', 'how', 'explain', 'tell me']):
            return random.choice(responses['question'])
        else:
            return random.choice(responses['default'])
