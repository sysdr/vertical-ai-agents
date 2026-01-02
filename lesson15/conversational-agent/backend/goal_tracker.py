from models import Goal, ConversationStateModel
import google.generativeai as genai

class GoalTracker:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def evaluate_goal_completion(self, goal: Goal, conversation_history: str) -> bool:
        if goal.completed:
            return True
        prompt = f"""Analyze if this goal has been achieved:
Goal: {goal.description}
Criteria: {goal.completion_criteria or 'User satisfaction'}
Conversation: {conversation_history}
Has the goal been achieved? Respond with only 'YES' or 'NO' and brief explanation."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().lower().startswith('yes')
        except:
            return False
    
    async def update_goals(self, state: ConversationStateModel) -> bool:
        if not state.active_goals:
            return False
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in state.messages[-10:]])
        any_completed = False
        for goal in state.active_goals:
            if not goal.completed:
                if await self.evaluate_goal_completion(goal, history):
                    state.complete_goal(goal.description)
                    any_completed = True
        return any_completed
    
    def get_active_goals_context(self, state: ConversationStateModel) -> str:
        active = [g for g in state.active_goals if not g.completed]
        if not active:
            return ""
        return "\nActive Goals:\n" + "\n".join([f"- {g.description}" for g in active])
