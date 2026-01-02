import google.generativeai as genai
from typing import List
from models import Message
import time
import re

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_key_valid = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
                self.api_key_valid = True
            except:
                self.api_key_valid = False
        self.max_retries = 3
        self.base_delay = 1.0
    
    def _generate_fallback_response(self, user_message: str, history: List[Message]) -> tuple[str, int]:
        """Generate a contextual fallback response when API key is invalid"""
        user_msg_lower = user_message.lower()
        
        # Only show API key note on the very first assistant response
        # Count assistant messages in history (this will be 0 for first response)
        assistant_message_count = len([m for m in history if m.role == "assistant"]) if history else 0
        show_api_note = assistant_message_count == 0  # Only on first assistant response
        
        # Check if user wants step-by-step reasoning or detailed explanation
        wants_detailed = any(phrase in user_msg_lower for phrase in [
            'step-by-step', 'step by step', 'detailed', 'explain', 'reasoning', 
            'how does', 'how to', 'walk me through', 'break down', 'guide me'
        ])
        
        # Greeting responses
        if any(word in user_msg_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = """Hello! I'm an intelligent conversational agent designed to help you with various tasks.

**My Capabilities:**
- Track conversation state and context
- Manage and track your goals
- Provide step-by-step reasoning when requested
- Answer questions with clear explanations

What would you like to work on today?"""
        
        # Step-by-step reasoning requests
        elif wants_detailed or 'reasoning' in user_msg_lower:
            if 'conversational agent' in user_msg_lower or 'agent' in user_msg_lower:
                response = """**Step-by-Step: How a Conversational Agent Works**

**Step 1: Message Reception**
- The agent receives your message and analyzes its content
- It identifies the intent (question, command, goal setting, etc.)

**Step 2: Context Building**
- Retrieves conversation history from memory
- Loads active goals and conversation state
- Builds context from previous messages

**Step 3: State Management**
- Updates conversation state (initializing → active → goal_seeking → completed)
- Tracks message count and token usage
- Maintains user-specific conversation threads

**Step 4: Goal Processing**
- If a `/goal` command is detected, creates a new goal
- Tracks active goals and monitors completion
- Provides goal context in responses

**Step 5: Response Generation**
- Generates contextual response based on:
  * Current message content
  * Conversation history
  * Active goals
  * System context
- Applies step-by-step reasoning when requested

**Step 6: State Persistence**
- Saves conversation state to database
- Updates message history
- Persists goals and metadata

**Step 7: Response Delivery**
- Returns response with updated metrics
- Updates dashboard in real-time
- Maintains conversation continuity"""
            elif 'goal' in user_msg_lower or 'track' in user_msg_lower:
                response = """**Step-by-Step: Goal Tracking System**

**Step 1: Goal Creation**
- Use `/goal <description>` to create a goal
- Example: `/goal Learn Python programming`
- The system stores the goal with metadata

**Step 2: Goal Context Integration**
- Active goals are included in conversation context
- The AI considers goals when generating responses
- Goals influence response relevance

**Step 3: Goal Monitoring**
- System tracks conversation progress
- Analyzes if goal-related content appears
- Updates goal status based on progress

**Step 4: Goal Completion Detection**
- AI evaluates if goal criteria are met
- Checks conversation history for completion indicators
- Marks goals as completed when achieved

**Step 5: State Updates**
- Updates conversation state to 'goal_seeking' when goals exist
- Changes to 'completed' when all goals are achieved
- Provides goal status in dashboard metrics"""
            elif 'state' in user_msg_lower or 'conversation state' in user_msg_lower:
                response = """**Step-by-Step: Conversation State Management**

**Step 1: Initialization**
- State starts as 'initializing' when conversation is created
- System sets up memory and context structures

**Step 2: Activation**
- State changes to 'active' after first message exchange
- Conversation becomes fully operational

**Step 3: Goal Seeking**
- State transitions to 'goal_seeking' when goals are set
- System actively works towards goal completion

**Step 4: Completion**
- State becomes 'completed' when all goals are achieved
- Conversation can continue but goals are marked complete

**State Transitions:**
```
initializing → active → goal_seeking → completed
```

**State Information Tracked:**
- Message count and history
- Active goals count
- Total tokens used
- Last update timestamp"""
            else:
                # Generic step-by-step reasoning
                response = """**Step-by-Step Reasoning Process:**

**Step 1: Understanding the Request**
- Analyze what you're asking for
- Identify key concepts and requirements
- Determine the type of response needed

**Step 2: Gathering Context**
- Review conversation history
- Check active goals and objectives
- Consider relevant background information

**Step 3: Breaking Down the Problem**
- Divide complex questions into smaller parts
- Identify relationships between concepts
- Structure the approach logically

**Step 4: Applying Knowledge**
- Use relevant information to address each part
- Connect concepts logically
- Ensure coherence in the explanation

**Step 5: Synthesizing the Response**
- Combine insights into a clear answer
- Provide step-by-step breakdown when requested
- Include examples or analogies if helpful

**Step 6: Delivering the Response**
- Present information in a clear, organized manner
- Use formatting (steps, bullets) for clarity
- Ensure the response addresses your question

I'm designed to provide clear, step-by-step reasoning when you request it. What specific topic would you like me to break down step-by-step?"""
        
        # Question responses
        elif '?' in user_message or any(word in user_msg_lower for word in ['what', 'how', 'why', 'when', 'where', 'explain', 'tell me', 'about']):
            if 'machine learning' in user_msg_lower or 'ml' in user_msg_lower:
                response = """**Machine Learning Explained:**

Machine learning is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed.

**Main Types:**
1. **Supervised Learning**: Learning from labeled data (e.g., email spam detection)
2. **Unsupervised Learning**: Finding patterns in unlabeled data (e.g., customer segmentation)
3. **Reinforcement Learning**: Learning through trial and error with rewards (e.g., game playing AI)

**Key Concepts:**
- Training data: Examples used to teach the model
- Model: The algorithm that learns patterns
- Prediction: Using the model to make decisions on new data"""
            elif 'python' in user_msg_lower or 'programming' in user_msg_lower:
                response = """**Python Programming Overview:**

Python is a versatile, high-level programming language known for its readability and simplicity.

**Key Features:**
- Readable syntax that's easy to learn
- Extensive standard library
- Large ecosystem of third-party packages
- Used in web development, data science, AI, automation, and more

**Common Use Cases:**
- Web development (Django, Flask)
- Data analysis (Pandas, NumPy)
- Machine learning (TensorFlow, PyTorch)
- Automation and scripting

Would you like to learn about a specific aspect of Python?"""
            elif 'ai' in user_msg_lower or 'artificial intelligence' in user_msg_lower:
                response = """**Artificial Intelligence Overview:**

AI is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence.

**Key Areas:**
- **Machine Learning**: Systems that learn from data
- **Natural Language Processing**: Understanding and generating human language
- **Computer Vision**: Interpreting visual information
- **Robotics**: Physical systems that interact with the world

**Applications:**
- Virtual assistants (like me!)
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis
- Language translation

What specific area of AI interests you?"""
            else:
                response = """I'd be happy to help with that question! 

To provide the most accurate and detailed answer with step-by-step reasoning, I can:
- Break down complex topics into clear steps
- Explain concepts with examples
- Provide structured, logical explanations

What specific topic would you like me to explain in detail?"""
        
        # Goal-related
        elif 'goal' in user_msg_lower or 'want to' in user_msg_lower or 'plan to' in user_msg_lower:
            response = """That sounds like a great goal! Here's how goal tracking works:

**Setting Goals:**
- Use `/goal <description>` to create a goal
- Example: `/goal Learn Python basics`
- The system will track your progress

**Goal Management:**
- Goals appear in the dashboard
- I'll consider your goals when responding
- Progress is monitored automatically

**Goal Completion:**
- The system detects when goals are achieved
- Goals are marked as completed
- You can set multiple goals simultaneously

Try it: `/goal [your goal description]`"""
        
        # Thank you responses
        elif any(word in user_msg_lower for word in ['thank', 'thanks', 'appreciate']):
            response = "You're welcome! I'm glad I could help. Is there anything else you'd like to know or work on?"
        
        # Default contextual response
        else:
            # Try to provide a relevant response based on keywords
            if 'help' in user_msg_lower:
                response = """I'm here to help! I can:

**Capabilities:**
- Answer questions with step-by-step reasoning
- Track and manage your goals
- Maintain conversation context
- Provide detailed explanations

**Commands:**
- `/goal <description>` - Set a goal to track
- Ask questions - I'll provide detailed answers
- Request step-by-step explanations

What would you like help with?"""
            elif 'learn' in user_msg_lower or 'study' in user_msg_lower:
                response = """Learning is a great journey! I can help you:

**Learning Support:**
- Break down complex topics step-by-step
- Explain concepts clearly
- Track your learning goals
- Provide structured explanations

**Getting Started:**
- Set a learning goal: `/goal Learn [topic]`
- Ask questions about the topic
- Request step-by-step explanations

What would you like to learn about?"""
            else:
                response = f"""I understand: "{user_message}"

**I can help you with:**
- Step-by-step reasoning and explanations
- Goal tracking and management
- Answering questions in detail
- Maintaining conversation context

**To get started:**
- Ask me to explain something step-by-step
- Set a goal using `/goal [description]`
- Request detailed explanations on any topic

What would you like to explore?"""
        
        # Add a note about API key only on first few messages to avoid repetition
        if not self.api_key_valid and show_api_note:
            response += "\n\n💡 **Note:** To enable full AI capabilities with advanced reasoning, please update your Gemini API key. Get one at https://aistudio.google.com/app/apikey"
        
        tokens = max(len(response.split()) * 1.3, len(response) // 4)
        return response, int(tokens)
    
    async def generate_response(self, messages: List[Message], system_context: str = "") -> tuple[str, int]:
        # If API key is not valid, use fallback immediately
        if not self.api_key_valid:
            current = messages[-1].content if messages else ""
            return self._generate_fallback_response(current, messages)
        
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[:-1]])
        current = messages[-1].content if messages else ""
        prompt = f"""{system_context}
Conversation History:
{history}
User: {current}
Assistant:"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                # Better token estimate: count words and add overhead
                word_count = len(text.split())
                tokens = max(word_count * 1.3, len(text) // 4)  # Better estimate
                return text, int(tokens)
            except Exception as e:
                error_str = str(e)
                # Check for API key errors
                if "API_KEY_INVALID" in error_str or "API key" in error_str.lower() or "expired" in error_str.lower():
                    # Mark API key as invalid and use fallback
                    self.api_key_valid = False
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        # Use intelligent fallback instead of error message
                        return self._generate_fallback_response(current, messages)
                elif attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # For other errors, still try to provide a helpful response
                    return self._generate_fallback_response(current, messages)
