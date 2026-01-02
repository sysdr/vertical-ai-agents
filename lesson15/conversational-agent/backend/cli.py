#!/usr/bin/env python3
import asyncio
from conversation_engine import ConversationEngine
from dotenv import load_dotenv
import os
import sys

load_dotenv()

async def main():
    API_KEY = os.getenv("GEMINI_API_KEY")
    engine = ConversationEngine(API_KEY)
    await engine.initialize()
    
    print("=== Conversational Agent CLI ===")
    print("Commands: /goal <description>, /quit")
    
    user_id = input("Enter your user ID: ").strip()
    conversation_id = await engine.create_conversation(user_id)
    print(f"Conversation started: {conversation_id}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
            
            result = await engine.process_message(conversation_id, user_input)
            print(f"\nAssistant: {result['response']}\n")
            print(f"[State: {result['state']}, Goals: {result['active_goals']}, Tokens: {result['total_tokens']}]\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
